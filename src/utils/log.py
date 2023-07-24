import logging
import os
from glob import glob

import hydra
from omegaconf import OmegaConf


class Logger:
    def __init__(self, cfg, wandb):
        self.cfg = cfg
        self.wandb = wandb

        self.id = None
        if cfg.train.resume:
            base_path = hydra.utils.get_original_cwd()
            try:
                self.id = glob(os.path.join(base_path, cfg.train.resume, 'wandb', 'run-*'))[0].rsplit('-', 1)[-1]
            except FileNotFoundError:
                print(f'Wandb folder is not founded at {os.path.join(base_path, cfg.train.resume)}')

        self._init_logger()

    def log(self, data):
        if self.logging:
            self.logging.info(data)
        if self.wandb_logger:
            self.wandb_logger.log(data)

    def _init_logger(self):
        if self.wandb:
            try:
                import wandb

                wandb.init(project=self.cfg.info.project, entity=self.cfg.info.entity,
                           config=OmegaConf.to_container(self.cfg), name=f"{self.cfg.name}", id=self.id,
                           settings=wandb.Settings(_disable_stats=True), save_code=True, resume='allow')
                # name=f"{datetime.now().strftime('%Y-%m-%d/%H:%M:%S')}/{self.cfg.name}",₩

                self.wandb_logger = wandb

            except (ImportError, AssertionError):
                prefix = 'Weights & Biases: '
                print(f"{prefix}run 'pip install wandb' to track and visualize.")

        self.logging = logging
