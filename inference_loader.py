import os

import hydra
import timm
from omegaconf import DictConfig
from timm.models import load_checkpoint

from src.data import get_dataloader
from src.fit import Fit
from src.initial_setting import init_seed, init_distributed, init_logger, cuda_setting
from src.models import *

methods = ['ADD', 'Sub']
architecture = ['33', '333', 'Inception', 'InceptionV2', 'InceptionV3']
weight_name = 'model_best.pth.tar'
weight_root = 'archive'


@hydra.main(config_path="configs", config_name="sub_resnet", version_base="1.3")
def main(cfg: DictConfig) -> None:
    import torch
    cfg.gpus = [0]
    cuda_setting(cfg.gpus)
    init_distributed(cfg)
    init_seed(cfg.train.seed + cfg.local_rank)

    device = torch.device(f'cuda:{cfg.local_rank}') if torch.cuda.is_available() else torch.device('cpu')
    cfg.model.model_name = 'resnet18_Sub333'
    init_logger(cfg)
    cfg.dataset.augmentation.prefetcher = True
    cfg.dataset.augmentation.pin_mem = True
    cfg.train.batch_size = 256
    cfg.dataset.size = [3, 256, 256]

    connectivity = f'{methods[1]}{architecture[3]}'
    model_name = f'resnet18_{connectivity}'
    model = timm.create_model(
        model_name,
        pretrained=False,
        in_chans=cfg.dataset.in_channels,
        num_classes=cfg.dataset.num_classes,
    )
    load_checkpoint(model, os.path.join(weight_root, connectivity, weight_name))
    model.eval()
    model = model.to(device)

    loaders = get_dataloader(cfg)

    import torch
    fit = Fit(cfg, None, device, (0, 0), model, torch.nn.CrossEntropyLoss(), None, None, None, None, None)
    model.eval()
    n_param = sum(p.numel() for p in fit.model.parameters() if p.requires_grad)
    with torch.no_grad():
        _, metrics = fit.test(False, loaders[-1])

    with torch.no_grad():
        _, metrics2 = fit.test(False, loaders[-1])

    deploy(fit.model, BasicBlockSub)
    with torch.no_grad():
        _, re_metrics = fit.test(False, loaders[-1])

    n_reparam = sum(p.numel() for p in fit.model.parameters() if p.requires_grad)

    print(n_param, n_reparam)
    print(metrics)
    print(metrics2)
    print(re_metrics)


if __name__ == "__main__":
    main()
