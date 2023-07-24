import torch
import hydra
import timm
from omegaconf import DictConfig

from src.data import get_dataloader
from src.fit.fit import FitForDyREP
from src.initial_setting import init_seed, init_distributed, init_logger, cuda_setting
from src.models import *
from src.models.dyrep_utils.dbb.dbb_block import SubDyREPDiverseBranchBlock, DyREPDiverseBranchBlock
from src.models.dyrep_utils.dyrep import DyRep
from src.models.dyrep_utils.recal_bn import recal_bn
from src.utils import model_tune, logging_benchmark_result_to_wandb, benchmark_model, ObjectFactory, CheckpointSaver


@hydra.main(config_path="configs", config_name="sub_imagenet", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cuda_setting(cfg.gpus)
    init_distributed(cfg)
    init_seed(cfg.train.seed + cfg.local_rank)

    device = torch.device(f'cuda:{cfg.local_rank}') if torch.cuda.is_available() else torch.device('cpu')

    init_logger(cfg)

    loaders = get_dataloader(cfg)
    factory = ObjectFactory(cfg)

    model = timm.create_model(
        **cfg.model,
        in_chans=cfg.dataset.in_channels,
        num_classes=cfg.dataset.num_classes,
    )
    model.to(device)

    if cfg.train.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if cfg.distributed and cfg.train.sync_bn:
        cfg.train.dist_bn = ''
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    optimizer, scheduler = factory.create_optimizer_and_scheduler(model, len(loaders[0]))
    criterion, scaler = factory.create_criterion_scaler()

    model, model_ema, start_epoch = model_tune(model, optimizer, scaler, scheduler, cfg)

    saver = CheckpointSaver(model=model, optimizer=optimizer, args=cfg, model_ema=model_ema, amp_scaler=scaler,
                            scheduler=scheduler, max_history=cfg.train.save_max_history)

    # if cfg.local_rank == 0 and cfg.wandb and not cfg.train.resume:
    #     benchmark_result = benchmark_model(cfg.benchmark, model)
    #     logging_benchmark_result_to_wandb(benchmark_result, cfg.name)

    dbb_block_fn = SubDyREPDiverseBranchBlock if 'sub' in cfg.name.lower() else DyREPDiverseBranchBlock
    dyrep = DyRep(
        model.module,
        optimizer,
        recal_bn_fn=lambda m: recal_bn(model.module, loaders[0], 200, m),
        filter_bias_and_bn=False,
        dbb_block_fn=dbb_block_fn)

    cfg = factory.cfg
    epochs = (start_epoch, cfg.train.epochs)
    fit = FitForDyREP(cfg, scaler, device, epochs, [model, dyrep], criterion, optimizer, model_ema, scheduler, saver,
                      loaders)
    fit()


if __name__ == "__main__":
    main()
