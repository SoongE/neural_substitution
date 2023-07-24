import copy
import os

import hydra
import pandas
import timm
from tqdm import tqdm

from src.data import get_dataloader
from src.models import *
from src.utils import clean_state_dict


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = clean_state_dict(checkpoint['state_dict'])
    model.load_state_dict(state_dict)


def forward(model, loader, device):
    pred = list()
    ys = list()
    for data in tqdm(loader, total=len(loader), leave=False):
        x, y = data[0].to(device), data[1].to(device)
        out = model(x)
        pred.append(out.argmax(-1))
        ys.append(y)
    return torch.concat(pred), torch.concat(ys)


@torch.no_grad()
def main():
    methods = ['ADD', 'Sub']
    architecture = ['33', '333', 'Inception', 'InceptionV2', 'InceptionV3']
    weight_name = 'model_best.pth.tar'
    weight_root = 'archive'

    with hydra.initialize(config_path='configs', version_base='1.3'):
        cfg = hydra.compose(config_name='sub_resnet', overrides=['local_rank=6', 'train.batch_size=256'])

    device = torch.device(f'cuda:{cfg.local_rank}') if torch.cuda.is_available() else torch.device('cpu')
    valid_loader = get_dataloader(cfg)[-1]

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
    # deploy(model, BasicBlockSub)

    pred, ys = forward(model, valid_loader, device)
    acc = pred.eq(ys).float().mean()
    df = pandas.DataFrame([['origin', acc.detach().cpu().item() * 100]])
    print("origin: ", acc)

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            _module_data = copy.deepcopy(module.weight.data)
            module.weight.data.fill_(0.0)

            pred, ys = forward(model, valid_loader, device)
            acc = pred.eq(ys).float().mean()
            df = pandas.concat([df, pandas.DataFrame([[name, acc.detach().cpu().item() * 100]])])
            print(name, acc)
            module.weight.data = _module_data

    df.columns = ['SkipConvName', 'Accuracy']
    print(df.describe())
    print(df)
    df.to_csv(f'assets/{model_name}.csv')


main()
