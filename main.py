import gc

import hydra
import torch
import wandb
from omegaconf import DictConfig

from src.data import get_dataloader
from src.fit import Fit
from src.initial_setting import init_seed, init_distributed, init_logger, cuda_setting
from src.utils import model_tune, ObjectFactory, CheckpointSaver


@hydra.main(config_path="configs", config_name="imagenet", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cuda_setting(cfg.gpus)
    init_distributed(cfg)
    init_seed(cfg.train.seed + cfg.local_rank)

    device = torch.device(f'cuda:{cfg.local_rank}') if torch.cuda.is_available() else torch.device('cpu')
    cfg.name = cfg.model.model_name if cfg.name == '' else cfg.name
    init_logger(cfg)

    loaders = get_dataloader(cfg)
    factory = ObjectFactory(cfg)

    model = factory.create_model()
    optimizer, scheduler = factory.create_optimizer_and_scheduler(cfg, model, len(loaders[0]))
    criterion, scaler = factory.create_criterion_scaler()

    model, model_ema, start_epoch, scheduler = model_tune(model, optimizer, scaler, scheduler, cfg)

    saver = CheckpointSaver(model=model, optimizer=optimizer, args=cfg, model_ema=model_ema, amp_scaler=scaler,
                            scheduler=scheduler, max_history=cfg.train.save_max_history)

    cfg = factory.cfg
    epochs = (start_epoch, cfg.train.epochs)
    fit = Fit(cfg, scaler, device, epochs, model, criterion, optimizer, model_ema, scheduler, saver, loaders)

    fit()

    if cfg.local_rank == 0:
        torch.cuda.empty_cache()
        gc.collect()
        wandb.finish(quiet=True)


if __name__ == "__main__":
    main()
