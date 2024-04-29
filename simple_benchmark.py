from time import perf_counter

import torch
from lightning_fabric.utilities import measure_flops, Throughput

from src.models import deploy, SubResNetStem
from src.models.resnet_sub import SubResNet

def format_human(size):
    power = 1000
    n = 0
    power_labels = {0: '', 1: 'K', 2: 'M', 3: 'G', 4: 'T'}
    while size > power:
        size /= power
        n += 1
    return f'{size:.1f} {power_labels[n]}'

def get_flops(model):
    model.eval()
    model_fwd = lambda: model(x)
    fwd_flops = measure_flops(model, model_fwd)
    return fwd_flops


def get_throughput(model):
    model.eval()
    throughput = Throughput(window_size=10)
    batch = xx.shape[0]
    t0 = perf_counter()
    for i in range(1, 11):
        _ = model(xx)
        throughput.update(time=perf_counter() - t0, samples=i * batch, batches=batch * i)
    return throughput.compute()['device/samples_per_sec']


with torch.device('meta'):
    origin_model = SubResNet('resnet50_origin')
    add_model = SubResNet('resnet50_HybV1')
    sub_model = SubResNetStem('resnet50_StemV1')
    x = torch.randn(1, 3, 224, 224)
    xx = torch.randn(512, 3, 224, 224)

origin_flops = get_flops(origin_model)
add_flops = get_flops(add_model)
sub_flops = get_flops(sub_model)

print(f'Origin | Add | Sub: {format_human(origin_flops)} | {format_human(add_flops)} | {format_human(sub_flops)}')
print(f'add/ori {add_flops / origin_flops:.2f}')
print(f'sub/ori {sub_flops / origin_flops:.2f}')
print(f'sub/add {sub_flops / add_flops:.2f}')

sub_throughput = get_throughput(sub_model)
add_throughput = get_throughput(add_model)
origin_throughput = get_throughput(origin_model)

print(f'Origin | Add | Sub: {origin_throughput:.1f} | {add_throughput:.1f} | {sub_throughput:.1f}')
print(f'add/ori {add_throughput / origin_throughput:.2f}')
print(f'sub/ori {sub_throughput / origin_throughput:.2f}')
print(f'sub/add {sub_throughput / add_throughput:.2f}')

deploy(sub_model)
sub_flops = get_flops(sub_model)
sub_throughput = get_throughput(sub_model)
print(f'Deploy Sub FLOPs: {format_human(sub_flops)}')
print(f'Deploy Sub Throughput: {sub_throughput:.2f}')

# from timm import create_model
# with torch.device('meta'):
#     resnet50 = create_model('resnet50', pretrained=False)
# _flops = get_flops(resnet50)
# _throughput = get_throughput(resnet50)
# print(f'FLOPs: {format_human(_flops)} / Throughput: {_throughput:.2f}')