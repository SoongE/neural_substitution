#!/usr/bin/env python3
""" Model Benchmark Script

An inference and train step benchmark script for timm models.

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import csv
import json
import logging
import time
from contextlib import suppress
from functools import partial

import torch
import torch.nn as nn
import torch.nn.parallel
from hydra import compose, initialize
from omegaconf import OmegaConf
from timm.data import resolve_data_config
from timm.models import create_model
from timm.utils import set_jit_fuser, decay_batch_step, check_batch_size_retry

try:
    from deepspeed.profiling.flops_profiler import get_model_profile

    has_deepspeed_profiling = True
except ImportError as e:
    has_deepspeed_profiling = False

try:
    from fvcore.nn import FlopCountAnalysis, flop_count_str, ActivationCountAnalysis

    has_fvcore_profiling = True
except ImportError as e:
    FlopCountAnalysis = None
    has_fvcore_profiling = False


def timestamp(sync=False):
    return time.perf_counter()


def cuda_timestamp(sync=False, device=None):
    if sync:
        torch.cuda.synchronize(device=device)
    return time.perf_counter()


def count_params(model: nn.Module):
    return sum([m.numel() for m in model.parameters()])


def resolve_precision(precision: str):
    assert precision in ('amp', 'float16', 'bfloat16', 'float32')
    use_amp = False
    model_dtype = torch.float32
    data_dtype = torch.float32
    if precision == 'amp':
        use_amp = True
    elif precision == 'float16':
        model_dtype = torch.float16
        data_dtype = torch.float16
    elif precision == 'bfloat16':
        model_dtype = torch.bfloat16
        data_dtype = torch.bfloat16
    return use_amp, model_dtype, data_dtype


def profile_deepspeed(model, input_size=(3, 224, 224), batch_size=1, detailed=False):
    _, macs, _ = get_model_profile(
        model=model,
        input_shape=(batch_size,) + input_size,  # input shape/resolution
        print_profile=detailed,  # prints the model graph with the measured profile attached to each module
        detailed=detailed,  # print the detailed profile
        warm_up=10,  # the number of warm-ups before measuring the time of each module
        as_string=False,  # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
        output_file=None,  # path to the output file. If None, the profiler prints to stdout.
        ignore_modules=None)  # the list of modules to ignore in the profiling
    return macs, 0  # no activation count in DS


def profile_fvcore(model, input_size=(3, 224, 224), batch_size=1, detailed=False, force_cpu=False):
    if force_cpu:
        model = model.to('cpu')
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    example_input = torch.ones((batch_size,) + input_size, device=device, dtype=dtype)
    fca = FlopCountAnalysis(model, example_input)
    aca = ActivationCountAnalysis(model, example_input)
    if detailed:
        fcs = flop_count_str(fca)
        print(fcs)
    return fca.total(), aca.total()


class BenchmarkRunner:
    def __init__(
            self,
            model_name,
            detail=False,
            device='cuda',
            torchscript=False,
            aot_autograd=False,
            precision='float32',
            fuser='',
            num_warm_iter=10,
            num_bench_iter=50,
            use_train_size=False,
            **kwargs
    ):
        self.model_name = model_name
        self.detail = detail
        self.device = device
        self.use_amp, self.model_dtype, self.data_dtype = resolve_precision(precision)
        self.channels_last = kwargs.pop('channels_last', False)
        self.amp_autocast = partial(torch.cuda.amp.autocast, dtype=torch.float16) if self.use_amp else suppress

        if fuser:
            set_jit_fuser(fuser)
        self.model = create_model(
            model_name,
            num_classes=kwargs.pop('num_classes', None),
            in_chans=3,
            global_pool=kwargs.pop('gp', 'fast'),
            scriptable=torchscript,
            drop_rate=kwargs.pop('drop', 0.),
            drop_path_rate=kwargs.pop('drop_path', None),
            drop_block_rate=kwargs.pop('drop_block', None),
        ) if isinstance(self.model_name, str) else self.model_name

        self.model.to(
            device=self.device,
            dtype=self.model_dtype,
            memory_format=torch.channels_last if self.channels_last else None)
        self.num_classes = kwargs.pop('num_classes')
        self.param_count = count_params(self.model)

        data_config = resolve_data_config(kwargs, model=self.model, use_test_size=not use_train_size)
        self.scripted = False
        if torchscript:
            self.model = torch.jit.script(self.model)
            self.scripted = True
        self.input_size = data_config['input_size']
        self.batch_size = kwargs.pop('batch_size', 256)

        self.example_inputs = None
        self.num_warm_iter = num_warm_iter
        self.num_bench_iter = num_bench_iter
        self.log_freq = num_bench_iter // 5
        if 'cuda' in self.device:
            self.time_fn = partial(cuda_timestamp, device=self.device)
        else:
            self.time_fn = timestamp

    def _init_input(self):
        self.example_inputs = torch.randn(
            (self.batch_size,) + self.input_size, device=self.device, dtype=self.data_dtype)
        if self.channels_last:
            self.example_inputs = self.example_inputs.contiguous(memory_format=torch.channels_last)


class InferenceBenchmarkRunner(BenchmarkRunner):
    def __init__(
            self,
            model_name,
            device='cuda',
            torchscript=False,
            **kwargs
    ):
        super().__init__(model_name=model_name, device=device, torchscript=torchscript, **kwargs)
        self.model.eval()

    def run(self):
        def _step():
            t_step_start = self.time_fn()
            with self.amp_autocast():
                output = self.model(self.example_inputs)
            t_step_end = self.time_fn(True)
            return t_step_end - t_step_start

        # _logger.info(
        #     f'Running inference benchmark on {self.model_name} for {self.num_bench_iter} steps w/ '
        #     f'input size {self.input_size} and batch size {self.batch_size}.')

        with torch.no_grad():
            self._init_input()

            for _ in range(self.num_warm_iter):
                _step()

            total_step = 0.
            num_samples = 0
            t_run_start = self.time_fn()
            for i in range(self.num_bench_iter):
                delta_fwd = _step()
                total_step += delta_fwd
                num_samples += self.batch_size
                num_steps = i + 1
                # if num_steps % self.log_freq == 0:
                # _logger.info(
                #     f"Infer [{num_steps}/{self.num_bench_iter}]."
                #     f" {num_samples / total_step:0.2f} samples/sec."
                #     f" {1000 * total_step / num_steps:0.3f} ms/step.")
            t_run_end = self.time_fn(True)
            t_run_elapsed = t_run_end - t_run_start

        results = dict(
            samples_per_sec=round(num_samples / t_run_elapsed, 2),
            step_time=round(1000 * total_step / self.num_bench_iter, 3),
            batch_size=self.batch_size,
            img_size=self.input_size[-1],
            param_count=round(self.param_count / 1e6, 2),
        )

        retries = 0 if self.scripted else 2  # skip profiling if model is scripted
        while retries:
            retries -= 1
            try:
                if has_deepspeed_profiling:
                    macs, _ = profile_deepspeed(self.model, self.input_size)
                    results['gmacs'] = round(macs / 1e9, 2)
                elif has_fvcore_profiling:
                    macs, activations = profile_fvcore(self.model, self.input_size, force_cpu=not retries)
                    results['gmacs'] = round(macs / 1e9, 2)
                    results['macts'] = round(activations / 1e6, 2)
            except RuntimeError as e:
                pass

        # _logger.info(
        #     f"Inference benchmark of {self.model_name} done. "
        #     f"{results['samples_per_sec']:.2f} samples/sec, {results['step_time']:.2f} ms/step")

        return results


class ProfileRunner(BenchmarkRunner):

    def __init__(self, model_name, device='cuda', profiler='', **kwargs):
        super().__init__(model_name=model_name, device=device, **kwargs)
        if not profiler:
            if has_deepspeed_profiling:
                profiler = 'deepspeed'
            elif has_fvcore_profiling:
                profiler = 'fvcore'
        assert profiler, "One of deepspeed or fvcore needs to be installed for profiling to work."
        self.profiler = profiler
        self.model.eval()

    def run(self):
        # _logger.info(
        #     f'Running profiler on {self.model_name} w/ '
        #     f'input size {self.input_size} and batch size {self.batch_size}.')

        macs = 0
        activations = 0
        if self.profiler == 'deepspeed':
            macs, _ = profile_deepspeed(self.model, self.input_size, batch_size=self.batch_size, detailed=True)
        elif self.profiler == 'fvcore':
            macs, activations = profile_fvcore(self.model, self.input_size, batch_size=self.batch_size, detailed=True)

        results = dict(
            gmacs=round(macs / 1e9, 2),
            macts=round(activations / 1e6, 2),
            batch_size=self.batch_size,
            img_size=self.input_size[-1],
            param_count=round(self.param_count / 1e6, 2),
        )

        # _logger.info(
        #     f"Profile of {self.model_name} done. "
        #     f"{results['gmacs']:.2f} GMACs, {results['param_count']:.2f} M params.")

        return results


def _try_run(
        model_name,
        bench_fn,
        bench_kwargs,
        initial_batch_size,
        no_batch_size_retry=False
):
    batch_size = initial_batch_size
    results = dict()
    error_str = 'Unknown'
    while batch_size:
        try:
            torch.cuda.empty_cache()
            bench = bench_fn(model_name=model_name, batch_size=batch_size, **bench_kwargs)
            results = bench.run()
            return results
        except RuntimeError as e:
            error_str = str(e)
            logging.error(f'"{error_str}" while running benchmark.')
            if not check_batch_size_retry(error_str):
                logging.error(f'Unrecoverable error encountered while benchmarking {model_name}, skipping.')
                break
            if no_batch_size_retry:
                break
        batch_size = decay_batch_step(batch_size)
        logging.warning(f'Reducing batch size to {batch_size} for retry.')
    results['error'] = error_str
    return results


def benchmark(args, model=None):
    if args.amp:
        args.precision = 'amp'
    # print(f'Benchmarking in {args.precision} precision. '
    #       f'{"NHWC" if args.channels_last else "NCHW"} layout. '
    #       f'torchscript {"enabled" if args.torchscript else "disabled"}')

    bench_kwargs = OmegaConf.to_container(args).copy()
    bench_kwargs.pop('amp')
    model_name = bench_kwargs.pop('model') if model is None else model
    batch_size = bench_kwargs.pop('batch_size')

    run_results = _try_run(
        model_name,
        InferenceBenchmarkRunner,
        bench_kwargs=bench_kwargs,
        initial_batch_size=batch_size,
        no_batch_size_retry=args.no_retry,
    )

    param_count = run_results.pop('infer_param_count', run_results.pop('train_param_count', 0))
    run_results.setdefault('param_count', param_count)

    return run_results


def main():
    with initialize('../../configs'):
        cfg = compose('base_set')
    results = benchmark(cfg.benchmark)
    print(f'--result\n{json.dumps(results, indent=4)}')


def benchmark_model(cfg, model):
    result = benchmark(cfg, model)
    result.pop('batch_size')
    result.pop('img_size')
    print(f'Benchmark result\n{json.dumps(result, indent=4)}')
    return result


def write_results(results_file, results):
    with open(results_file, mode='w') as cf:
        dw = csv.DictWriter(cf, fieldnames=results[0].keys())
        dw.writeheader()
        for r in results:
            dw.writerow(r)
        cf.flush()


if __name__ == '__main__':
    with initialize('../../configs'):
        cfg = compose('config')

    model = create_model('resnet50')
    results = benchmark_model(cfg.set.benchmark, model)
    print(results)
