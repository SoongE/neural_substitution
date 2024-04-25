from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

from src.data import get_dataloader
from src.fit import Fit
from src.models import deploy
from src.utils import ObjectFactory

device = torch.device('cuda:0')
root = 'runs/cifar100_resnet50_HybV44/resnet50_HybV44_20240425-070418'
root = Path(root)

with open(root / 'summary.csv', 'r') as f:
    best_accuracy = float(f.readlines()[-1].split(',')[-1])

with hydra.initialize(config_path=str(root / '.hydra'), version_base='1.3'):
    cfg = hydra.compose(
        config_name='config',
        overrides=OmegaConf.load(str(root / '.hydra' / 'overrides.yaml')),
    )

factory = ObjectFactory(cfg)
model = factory.create_model()
state_dict = torch.load(root / 'model_best.pth.tar', map_location='cpu')['state_dict']
model.load_state_dict(state_dict)

loaders = get_dataloader(cfg)

fit = Fit(cfg, True, device, (0, 1), model, torch.nn.CrossEntropyLoss(), None, False, None, None, loaders)

eval_metrics = fit.validate(cfg.train.epochs)
print(best_accuracy, eval_metrics['Top1'])

deploy(model)
deploy_eval_metrics = fit.validate(cfg.train.epochs)
print(eval_metrics['Top1'], deploy_eval_metrics['Top1'])
