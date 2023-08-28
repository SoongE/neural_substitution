""" Model Benchmark Script
An inference and train step benchmark script for timm models.
Hacked together by Ross Wightman (https://github.com/rwightman)
"""

import csv
import json
import os
import time
from collections import OrderedDict
from contextlib import suppress
from copy import copy
from functools import partial

import hydra
import torch
import torch.nn as nn
import torch.nn.parallel
from deepspeed.profiling.flops_profiler import get_model_profile
from omegaconf import OmegaConf, ListConfig
from timm.data import resolve_data_config
from timm.models import create_model
from timm.optim import create_optimizer_v2

from src.models import SubResNet, AddResNet, SubMobileNet, AddMobileNet
from src.models.mobileone import mobileone

torch.backends.cudnn.benchmark = True


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
        # input_res for lower version
        # input_shape for higher version
        input_shape=(batch_size,) + input_size,  # input shape or input to the input_constructor
        print_profile=detailed,  # prints the model graph with the measured profile attached to each module
        detailed=detailed,  # print the detailed profile
        warm_up=10,  # the number of warm-ups before measuring the time of each module
        as_string=False,  # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
        output_file=None,  # path to the output file. If None, the profiler prints to stdout.
        ignore_modules=None)  # the list of modules to ignore in the profiling
    return macs, 0  # no activation count in DS


def profile_fvcore(model, input_size=(3, 224, 224), batch_size=1, detailed=False, force_cpu=False):
    pass
    # if force_cpu:
    #     model = model.to('cpu')
    # device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    # example_input = torch.ones((batch_size,) + input_size, device=device, dtype=dtype)
    # fca = FlopCountAnalysis(model, example_input)
    # aca = ActivationCountAnalysis(model, example_input)
    # if detailed:
    #     fcs = flop_count_str(fca)
    #     print(fcs)
    # return fca.total(), aca.total()


def create_model_cls(model_name, in_channels, num_classes, **kwargs):
    kwargs['in_channels'] = in_channels
    kwargs['num_classes'] = num_classes

    if 'resnet' in model_name or 'resnext' in model_name:
        if 'Sub' in model_name:
            model_cls = SubResNet
        else:
            model_cls = AddResNet
    elif 'mobilenet' in model_name:
        if 'Sub' in model_name:
            model_cls = SubMobileNet
        else:
            model_cls = AddMobileNet
    elif 'mobileone' in model_name:
        if 'Sub' in model_name:
            model_cls = mobileone
            kwargs.update({'substitution': True})
        else:
            model_cls = mobileone
            kwargs.update({'substitution': False})
    else:
        raise NotImplementedError(f'{model_name} is not implemented')
    return model_cls(model_name, **kwargs)



class BenchmarkRunner:
    def __init__(
            self, model_name, detail=False, device='cuda', torchscript=False, precision='float32',
            num_warm_iter=10, num_bench_iter=50, use_train_size=False, **kwcfg):
        self.model_name = model_name
        self.detail = detail
        self.device = device
        self.use_amp, self.model_dtype, self.data_dtype = resolve_precision(precision)
        self.channels_last = kwcfg.pop('channels_last', False)
        self.amp_autocast = torch.cuda.amp.autocast if self.use_amp else suppress

        self.model = create_model_cls(
            model_name,
            num_classes=kwcfg.pop('num_classes', None),
            in_channels=3,
            global_pool=kwcfg.pop('gp', 'fast'),
            scriptable=torchscript,
        )
        self.model.to(
            device=self.device,
            dtype=self.model_dtype,
            memory_format=torch.channels_last if self.channels_last else None)
        self.num_classes = self.model.num_classes
        self.param_count = count_params(self.model)
        print('Model %s created, param count: %d' % (model_name, self.param_count))
        self.scripted = False
        if torchscript:
            self.model = torch.jit.script(self.model)
            self.scripted = True

        data_config = resolve_data_config(kwcfg, model=self.model, use_test_size=not use_train_size)
        self.input_size = data_config['input_size']
        self.batch_size = kwcfg.pop('batch_size', 256)

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
    def __init__(self, model_name, device='cuda', torchscript=False, **kwcfg):
        super().__init__(model_name=model_name, device=device, torchscript=torchscript, **kwcfg)
        self.model.eval()

    def run(self):
        def _step():
            t_step_start = self.time_fn()
            with self.amp_autocast():
                output = self.model(self.example_inputs)
            t_step_end = self.time_fn(True)
            return t_step_end - t_step_start

        print(
            f'Running inference benchmark on {self.model_name} for {self.num_bench_iter} steps w/ '
            f'input size {self.input_size} and batch size {self.batch_size}.')

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
                if num_steps % self.log_freq == 0:
                    print(
                        f"Infer [{num_steps}/{self.num_bench_iter}]."
                        f" {num_samples / total_step:0.2f} samples/sec."
                        f" {1000 * total_step / num_steps:0.3f} ms/step.")
            t_run_end = self.time_fn(True)
            t_run_elapsed = t_run_end - t_run_start

        results = dict(
            samples_per_sec=round(num_samples / t_run_elapsed, 2),
            step_time=round(1000 * total_step / self.num_bench_iter, 3),
            batch_size=self.batch_size,
            img_size=self.input_size[-1],
            param_count=round(self.param_count / 1e6, 2),
            max_memory=torch.cuda.max_memory_allocated(device='cuda:0') / (1024 * 1024 * 1024),
        )
        torch.cuda.reset_peak_memory_stats()

        retries = 0 if self.scripted else 2  # skip profiling if model is scripted
        while retries:
            retries -= 1
            try:
                macs, _, = profile_deepspeed(self.model, self.input_size)
                results['gmacs'] = round(macs / 1e9, 2)
            except RuntimeError as e:
                pass

        print(
            f"Inference benchmark of {self.model_name} done. "
            f"{results['samples_per_sec']:.2f} samples/sec, {results['step_time']:.2f} ms/step")

        return results


class TrainBenchmarkRunner(BenchmarkRunner):
    def __init__(self, model_name, device='cuda', torchscript=False, **kwcfg):
        super().__init__(model_name=model_name, device=device, torchscript=torchscript, **kwcfg)
        self.model.train()

        if kwcfg.pop('smoothing', 0) > 0:
            self.loss = nn.CrossEntropyLoss().to(self.device)
        else:
            self.loss = nn.CrossEntropyLoss().to(self.device)
        self.target_shape = tuple()

        self.optimizer = create_optimizer_v2(
            self.model,
            opt=kwcfg.pop('opt', 'sgd'),
            lr=kwcfg.pop('lr', 1e-4))

    def _gen_target(self, batch_size):
        return torch.empty(
            (batch_size,) + self.target_shape, device=self.device, dtype=torch.long).random_(self.num_classes)

    def run(self):
        def _step(detail=False):
            self.optimizer.zero_grad()  # can this be ignored?
            t_start = self.time_fn()
            t_fwd_end = t_start
            t_bwd_end = t_start
            with self.amp_autocast():
                output = self.model(self.example_inputs)
                if isinstance(output, tuple):
                    output = output[0]
                if detail:
                    t_fwd_end = self.time_fn(True)
                target = self._gen_target(output.shape[0])
                self.loss(output, target).backward()
                if detail:
                    t_bwd_end = self.time_fn(True)
            self.optimizer.step()
            t_end = self.time_fn(True)
            if detail:
                delta_fwd = t_fwd_end - t_start
                delta_bwd = t_bwd_end - t_fwd_end
                delta_opt = t_end - t_bwd_end
                return delta_fwd, delta_bwd, delta_opt
            else:
                delta_step = t_end - t_start
                return delta_step

        print(
            f'Running train benchmark on {self.model_name} for {self.num_bench_iter} steps w/ '
            f'input size {self.input_size} and batch size {self.batch_size}.')

        self._init_input()

        for _ in range(self.num_warm_iter):
            _step()

        t_run_start = self.time_fn()
        if self.detail:
            total_fwd = 0.
            total_bwd = 0.
            total_opt = 0.
            num_samples = 0
            for i in range(self.num_bench_iter):
                delta_fwd, delta_bwd, delta_opt = _step(True)
                num_samples += self.batch_size
                total_fwd += delta_fwd
                total_bwd += delta_bwd
                total_opt += delta_opt
                num_steps = (i + 1)
                if num_steps % self.log_freq == 0:
                    total_step = total_fwd + total_bwd + total_opt
                    print(
                        f"Train [{num_steps}/{self.num_bench_iter}]."
                        f" {num_samples / total_step:0.2f} samples/sec."
                        f" {1000 * total_fwd / num_steps:0.3f} ms/step fwd,"
                        f" {1000 * total_bwd / num_steps:0.3f} ms/step bwd,"
                        f" {1000 * total_opt / num_steps:0.3f} ms/step opt."
                    )
            total_step = total_fwd + total_bwd + total_opt
            t_run_elapsed = self.time_fn() - t_run_start
            results = dict(
                samples_per_sec=round(num_samples / t_run_elapsed, 2),
                step_time=round(1000 * total_step / self.num_bench_iter, 3),
                fwd_time=round(1000 * total_fwd / self.num_bench_iter, 3),
                bwd_time=round(1000 * total_bwd / self.num_bench_iter, 3),
                opt_time=round(1000 * total_opt / self.num_bench_iter, 3),
                batch_size=self.batch_size,
                img_size=self.input_size[-1],
                param_count=round(self.param_count / 1e6, 2),
            )
        else:
            total_step = 0.
            num_samples = 0
            for i in range(self.num_bench_iter):
                delta_step = _step(False)
                num_samples += self.batch_size
                total_step += delta_step
                num_steps = (i + 1)
                if num_steps % self.log_freq == 0:
                    print(
                        f"Train [{num_steps}/{self.num_bench_iter}]."
                        f" {num_samples / total_step:0.2f} samples/sec."
                        f" {1000 * total_step / num_steps:0.3f} ms/step.")
            t_run_elapsed = self.time_fn() - t_run_start
            results = dict(
                samples_per_sec=round(num_samples / t_run_elapsed, 2),
                step_time=round(1000 * total_step / self.num_bench_iter, 3),
                batch_size=self.batch_size,
                img_size=self.input_size[-1],
                param_count=round(self.param_count / 1e6, 2),
            )

        print(
            f"Train benchmark of {self.model_name} done. "
            f"{results['samples_per_sec']:.2f} samples/sec, {results['step_time']:.2f} ms/sample")

        return results


class ProfileRunner(BenchmarkRunner):
    def __init__(self, model_name, device='cuda', profiler='', **kwcfg):
        super().__init__(model_name=model_name, device=device, **kwcfg)
        assert profiler, "One of deepspeed or fvcore needs to be installed for profiling to work."
        self.batch_size = 1
        self.profiler = profiler
        self.model.eval()

    def run(self):
        print(
            f'Running profiler on {self.model_name} w/ '
            f'input size {self.input_size} and batch size {self.batch_size}.')

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

        print(
            f"Profile of {self.model_name} done. "
            f"{results['gmacs']:.2f} GMACs, {results['param_count']:.2f} M params.")

        return results


def decay_batch_exp(batch_size, factor=0.5, divisor=16):
    out_batch_size = batch_size * factor
    if out_batch_size > divisor:
        out_batch_size = (out_batch_size + 1) // divisor * divisor
    else:
        out_batch_size = batch_size - 1
    return max(0, int(out_batch_size))


def _try_run(model_name, bench_fn, initial_batch_size, bench_kwcfg):
    batch_size = initial_batch_size
    results = dict()
    while batch_size >= 1:
        # bench = bench_fn(model_name=model_name, batch_size=batch_size, **bench_kwcfg)
        # results = bench.run()
        # return results
        try:
            bench = bench_fn(model_name=model_name, batch_size=batch_size, **bench_kwcfg)
            results = bench.run()
            return results
        except RuntimeError as e:
            e_str = str(e)
            print(e_str)
            if 'channels_last' in e_str:
                print(f'Error: {model_name} not supported in channels_last, skipping.')
                break
            print(f'Error: "{e_str}" while running benchmark. Reducing batch size to {batch_size} for retry.')
        batch_size = decay_batch_exp(batch_size)
    return results


def benchmark(cfg):
    if cfg.amp and cfg.precision != 'amp':
        print("Overriding precision to 'amp' since --amp flag set.")
        cfg.precision = 'amp'
    print(f'Benchmarking in {cfg.precision} precision. '
          f'{"NHWC" if cfg.channels_last else "NCHW"} layout. '
          f'torchscript {"enabled" if cfg.torchscript else "disabled"}')

    bench_kwcfg = copy(OmegaConf.to_container(cfg))

    bench_kwcfg.pop('amp')
    model = bench_kwcfg.pop('model')
    batch_size = bench_kwcfg.pop('batch_size')

    bench_fns = (InferenceBenchmarkRunner,)
    prefixes = ('infer',)
    if cfg.bench == 'both':
        bench_fns = (
            InferenceBenchmarkRunner,
            TrainBenchmarkRunner,
        )
        prefixes = ('infer', 'train')
    elif cfg.bench == 'all':
        bench_fns = (
            InferenceBenchmarkRunner,
            TrainBenchmarkRunner,
            ProfileRunner,
        )
        bench_kwcfg['profiler'] = 'deepspeed'
        prefixes = ('infer', 'train', 'profiler')
    elif cfg.bench == 'train':
        bench_fns = TrainBenchmarkRunner,
        prefixes = 'train',
    elif cfg.bench.startswith('profile'):
        # specific profiler used if included in bench mode string, otherwise default to deepspeed, fallback to fvcore
        bench_kwcfg['profiler'] = 'deepspeed'
        bench_fns = ProfileRunner,
        batch_size = 1

    model_results = OrderedDict(model=model)
    for prefix, bench_fn in zip(prefixes, bench_fns):
        run_results = _try_run(model, bench_fn, initial_batch_size=batch_size, bench_kwcfg=bench_kwcfg)
        if prefix:
            run_results = {'_'.join([prefix, k]): v for k, v in run_results.items()}
        model_results.update(run_results)
    param_count = model_results.pop('infer_param_count', model_results.pop('train_param_count', 0))
    model_results.setdefault('param_count', param_count)
    model_results.pop('train_param_count', 0)
    return model_results if model_results['param_count'] else dict()


def write_results(result_file, results):
    with open(result_file, mode='w') as cf:
        dw = csv.DictWriter(cf, fieldnames=results[0].keys())
        dw.writeheader()
        for r in results:
            dw.writerow(r)
        cf.flush()


@hydra.main(config_path="configs/benchmark", config_name="benchmark", version_base="1.3")
def main(cfg) -> None:
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpus)

    if isinstance(cfg.model, (list, ListConfig)):
        model_list = list(cfg.model)
    else:
        model_list = [cfg.model]

    print(f'Running bulk validation on these pretrained models: {model_list}')
    results = []

    try:
        for model in model_list:
            cfg.model = model
            result = benchmark(cfg)
            if result:
                results.append(result)
            time.sleep(10)
    except KeyboardInterrupt as e:
        pass

    sort_key = 'infer_samples_per_sec'
    if 'train' in cfg.bench:
        sort_key = 'train_samples_per_sec'
    elif 'profile' in cfg.bench:
        sort_key = 'infer_gmacs'
    results = sorted(results, key=lambda x: x[sort_key], reverse=True)

    if len(results):
        write_results(cfg.result_file, results)

    json_str = json.dumps(results, indent=4)
    print(json_str)


if __name__ == '__main__':
    main()
