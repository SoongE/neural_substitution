import copy
import gc
import os
from functools import reduce

import hydra
import pandas
import torch
from torch import nn
from tqdm import tqdm

from src.data import get_dataloader
from src.utils import clean_state_dict
from src.utils.factory import create_model_cls

gpus = [3]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in gpus)

weight_root = 'weights'


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr):
    return reduce(getattr, [obj] + attr.split('.'))


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
def main(cfg, backbone, method, architecture):
    device = torch.device(f'cuda:{cfg.local_rank}') if torch.cuda.is_available() else torch.device('cpu')
    valid_loader = get_dataloader(cfg)[-1]

    connectivity = f'{method}{architecture}'
    model_name = f'{backbone}_{connectivity}'
    model = create_model_cls(
        model_name,
        in_channels=cfg.dataset.in_channels,
        num_classes=cfg.dataset.num_classes,
        stem_type=cfg.dataset.name,
    )
    load_checkpoint(model,
                    os.path.join(weight_root, cfg.dataset.name, model_name.split('_')[0], model_name + '.pth.tar'))
    model.eval()
    model.to(device)
    # deploy(model, BasicBlockSub)

    pred, ys = forward(model, valid_loader, device)
    acc = pred.eq(ys).float().mean()
    df = pandas.DataFrame([['origin', acc.detach().cpu().item() * 100]])
    print("origin: ", acc.detach().cpu().item() * 100)
    model.to('cpu')

    drop_layers = list()
    for name, module in model.named_modules():
        if 'downsample' in name:
            continue
        if isinstance(module, torch.nn.Conv2d):
            if 'conv2' in name and module.stride == (1, 1):
                drop_layers.append((name, module))

    for i, (name, module) in enumerate(drop_layers):
        print(f'[{i}/{len(drop_layers)}] {name}')
        drop_model = copy.deepcopy(model)
        # module.weight.data.fill_(0.0)
        # rsetattr(drop_model, name, module)
        rsetattr(drop_model, name, nn.Identity())

        drop_model.to(device)
        pred, ys = forward(drop_model, valid_loader, device)
        acc = pred.eq(ys).float().mean()
        df = pandas.concat([df, pandas.DataFrame([[name, acc.detach().cpu().item() * 100]])])

        drop_model.to('cpu')
        del drop_model
        torch.cuda.empty_cache()
        gc.collect()

    df.columns = ['SkipConvName', 'Accuracy']
    print(df.describe())
    print(df)
    df.to_csv(f'analysis_droppath_{model_name}.csv')


if __name__ == '__main__':
    # methods = ['Add', 'Sub']
    # architecture = ['33', '333', 'InceptionV1', 'InceptionV2', 'InceptionV3']

    with hydra.initialize(config_path='configs', version_base='1.3'):
        cfg = hydra.compose(config_name='sub_cifar', overrides=['local_rank=0', 'train.batch_size=400'])

    main(cfg, backbone='resnet50', method='Add', architecture='InceptionV3')
