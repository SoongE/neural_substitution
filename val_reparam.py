import time
from datetime import timedelta

import torch
import torchmetrics
from rich.console import Console
from rich.progress import track
from rich.table import Table
from timm.data import create_transform
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet

from src.models import deploy, SubResNet
from src.utils.metadata import count_parameters, format_decimal

if __name__ == '__main__':
    model_name = 'resnet50_NS'
    dataset_name = 'imagenet'
    imagenet = ImageNet('/data/imageNet', split='val', transform=create_transform(224, crop_pct=1.0))
    imagenet = DataLoader(imagenet, batch_size=64, shuffle=False, num_workers=4)

    device = torch.device('cuda:0')
    num_classes = 1000
    stem_type = 'imagenet'
    loader = imagenet
    example = torch.rand(2, 3, 224, 224)

    # declare metric
    top1, top5 = torchmetrics.Accuracy('multiclass', num_classes=num_classes, top_k=1), torchmetrics.Accuracy(
        'multiclass', num_classes=num_classes, top_k=5)
    metric = torchmetrics.MetricCollection({'top1': top1, 'top5': top5})
    metric = metric.to(device)

    # declare model
    model = SubResNet(model_name, num_classes=num_classes, stem_type=stem_type)
    model.load_state_dict(torch.load(f'{model_name}.pth', map_location='cpu'))
    model.to(device)
    model.eval()

    # run evaluation before re-param
    s = time.perf_counter()
    for item in track(loader, description=f'Before Re-param {model_name} on {dataset_name}'):
        x, y = map(lambda x: x.to(device), item)
        with torch.cuda.amp.autocast() and torch.no_grad():
            out = model(x)
        metric.update(out, y)

    naive_runtime = time.perf_counter() - s
    naive_accuracy = metric.compute()
    naive_parameters = count_parameters(model)
    print(f'Before Re-param: {naive_accuracy} / {count_parameters(model)}')

    deploy(model)
    model.eval()
    metric.reset()
    s = time.perf_counter()
    for item in track(loader, description=f'After Re-param {model_name} on {dataset_name}'):
        x, y = map(lambda x: x.to(device), item)
        with torch.cuda.amp.autocast() and torch.no_grad():
            out = model(x)
        metric.update(out, y)

    deploy_runtime = time.perf_counter() - s
    deploy_accuracy = metric.compute()
    deploy_parameters = count_parameters(model)
    print(f'After Re-param: {deploy_accuracy} / {count_parameters(model)}')

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column('ReParam', justify='center')
    table.add_column('Top1 Accuracy', justify='center')
    table.add_column('# of Parameters', justify='center')
    table.add_column('Running Time', justify='center')

    table.add_row(*['No', str(round(naive_accuracy['top1'].item() * 100, 1)) + ' %', format_decimal(naive_parameters), timedelta(seconds=naive_runtime).__str__()])
    table.add_row(*['Yes', str(round(deploy_accuracy['top1'].item() * 100, 1)) + ' %', format_decimal(deploy_parameters), timedelta(seconds=deploy_runtime).__str__()])
    Console().print(table)
