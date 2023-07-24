import logging
import os
from collections import OrderedDict

import hydra
import torch
import wandb
from timm.models import load_checkpoint
from timm.utils import ModelEmaV2
from torch.nn.parallel import DistributedDataParallel


def print_pass(*args, **kwargs):
    pass


def model_tune(model, optimizer, loss_scaler, scheduler, cfg):
    resume = cfg.train.resume
    start_epoch = 0
    if resume:
        start_epoch = resume_checkpoint(model, resume, optimizer, loss_scaler, scheduler, cfg.local_rank == 0)
        scheduler.step(start_epoch)

    model_ema = ModelEmaV2(model, decay=cfg.train.model_ema_decay,
                           device='cpu' if cfg.model_ema_force_cpu else None) if cfg.train.model_ema else None
    load_checkpoint(model_ema.module, resume, use_ema=cfg.train.model_ema) if resume and model_ema else None

    if cfg.distributed:
        model = DistributedDataParallel(model, static_graph=False, device_ids=[cfg.local_rank],
                                        broadcast_buffers=cfg.train.ddp_bb)
    if cfg.torchcompile:
        model = torch.compile(model, backend='inductor')
    return model, model_ema, start_epoch, scheduler


def logging_benchmark_result_to_wandb(benchmark_result, exp_name):
    columns = ['Name']
    columns.extend(list(benchmark_result.keys()))
    data = [exp_name]
    data.extend(list(benchmark_result.values()))
    wandb.log({'benchmark': wandb.Table(columns=columns, data=[data])})


def resume_checkpoint(model, checkpoint_path, optimizer=None, loss_scaler=None, scheduler=None, log_info=True):
    resume_epoch = None
    base_path = hydra.utils.get_original_cwd()
    checkpoint_path = os.path.join(base_path, checkpoint_path, 'last.pth.tar')

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            if log_info:
                logging.info('Restoring model state from checkpoint...')
            state_dict = clean_state_dict(checkpoint['state_dict'])
            model.load_state_dict(state_dict)

            if optimizer is not None and 'optimizer' in checkpoint:
                if log_info:
                    logging.info('Restoring optimizer state from checkpoint...')
                optimizer.load_state_dict(checkpoint['optimizer'])

            if loss_scaler is not None and loss_scaler.state_dict_key in checkpoint:
                if log_info:
                    logging.info('Restoring AMP loss scaler state from checkpoint...')
                loss_scaler.load_state_dict(checkpoint[loss_scaler.state_dict_key])

            if scheduler is not None and 'scheduler' in checkpoint:
                if log_info:
                    logging.info('Restoring scheduler state from checkpoint...')
                scheduler.load_state_dict(checkpoint['scheduler'])

            if 'epoch' in checkpoint:
                resume_epoch = checkpoint['epoch']
                if 'version' in checkpoint and checkpoint['version'] > 1:
                    resume_epoch += 1  # start at the next epoch, old checkpoints incremented before save

            if log_info:
                logging.info("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
        else:
            model.load_state_dict(checkpoint)
            if log_info:
                logging.info("Loaded checkpoint '{}'".format(checkpoint_path))
        return resume_epoch
    else:
        logging.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def clean_state_dict(state_dict):
    # 'clean' checkpoint by removing .module prefix from state dict if it exists from parallel training
    cleaned_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        cleaned_state_dict[name] = v
    return cleaned_state_dict
