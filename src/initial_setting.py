import builtins
import os
import random

import numpy as np
import torch
from torch.distributed import init_process_group

from src.utils import Logger, print_pass


def init_logger(cfg):
    if cfg.is_master:
        logger = Logger(cfg, cfg.wandb)
        return logger


def init_distributed(cfg):
    try:
        cfg.world_size = int(os.environ['WORLD_SIZE'])
    except:
        cfg.world_size = 1
    cfg.distributed = cfg.world_size > 1

    if cfg.distributed:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend='nccl',
                init_method='env://',
            )

        cfg.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        cfg.world_size = torch.distributed.get_world_size()
        torch.cuda.set_device(cfg.local_rank)
        torch.cuda.empty_cache()

        if cfg.local_rank != 0:
            builtins.print = print_pass

        cfg.is_master = cfg.local_rank == 0


def cuda_setting(gpus):
    if isinstance(gpus, int):
        gpus = [gpus]
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in gpus)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True # If deterministic = True, training time will be increased.


def init_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
