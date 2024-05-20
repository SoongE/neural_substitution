from time import perf_counter

import torch
from lightning_fabric.utilities import measure_flops, Throughput

from src.models import SubMobileNet, AddMobileNet, AddResNet, SubResNetStem, SubResNet


def format_human(size):
    power = 1000
    n = 0
    power_labels = {0: '', 1: 'K', 2: 'M', 3: 'G', 4: 'T'}
    while size > power:
        size /= power
        n += 1
    return f'{size:.1f} {power_labels[n]}'


def get_flops(model):
    model.train()
    model_fwd = lambda: model(x)
    fwd_flops = measure_flops(model, model_fwd)
    return fwd_flops


def get_throughput(model):
    model.train()
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    throughput = Throughput(window_size=10)
    batch = xx.shape[0]
    t0 = perf_counter()
    for i in range(1, 11):
        # optim.zero_grad()
        out = model(xx)
        loss = criterion(out, yy)
        # loss.backward()
        # optim.step()
        throughput.update(time=perf_counter() - t0, samples=i * batch, batches=batch * i)
    return throughput.compute()['device/samples_per_sec']


# with torch.device('meta'):
#     origin_model = SubResNet('resnet50_origin')
#     add_model = SubResNet('resnet50_AddV1')
#     sub_model = SubResNetStem('resnet50_StemV4')
#     x = torch.randn(1, 3, 224, 224)
#     xx = torch.randn(512, 3, 224, 224)
#
# origin_flops = get_flops(origin_model)
# add_flops = get_flops(add_model)
# sub_flops = get_flops(sub_model)
#
# print(f'Origin | Add | Sub: {format_human(origin_flops)} | {format_human(add_flops)} | {format_human(sub_flops)}')
# print(f'add/ori {add_flops / origin_flops:.2f}')
# print(f'sub/ori {sub_flops / origin_flops:.2f}')
# print(f'sub/add {sub_flops / add_flops:.2f}')
#
# sub_throughput = get_throughput(sub_model)
# add_throughput = get_throughput(add_model)
# origin_throughput = get_throughput(origin_model)
#
# print(f'Origin | Add | Sub: {origin_throughput:.1f} | {add_throughput:.1f} | {sub_throughput:.1f}')
# print(f'add/ori {add_throughput / origin_throughput:.2f}')
# print(f'sub/ori {sub_throughput / origin_throughput:.2f}')
# print(f'sub/add {sub_throughput / add_throughput:.2f}')
#
# deploy(sub_model)
# sub_flops = get_flops(sub_model)
# sub_throughput = get_throughput(sub_model)
# print(f'Deploy Sub FLOPs: {format_human(sub_flops)}')
# print(f'Deploy Sub Throughput: {sub_throughput:.2f}')
network_fn = AddResNet
subnetwork_fn = SubResNetStem
backbone = 'resnet50'
with torch.device('meta'):
    origin_model = network_fn(f'{backbone}_origin', stem_type='imagenet')
    dbb_model = network_fn(f'{backbone}_AddV1', stem_type='imagenet')
    acnet_model = network_fn(f'{backbone}_AddV2', stem_type='imagenet')
    acnetp_model = network_fn(f'{backbone}_AddV3', stem_type='imagenet')
    sub_model = subnetwork_fn(f'{backbone}_StemV7C', stem_type='imagenet')
    x = torch.randn(1, 3, 224, 224)
    xx = torch.randn(512, 3, 224, 224)
    yy = torch.ones(512, dtype=torch.long)
    criterion = torch.nn.CrossEntropyLoss()
    models = [origin_model, sub_model, dbb_model, acnet_model, acnetp_model]
    names = ['origin', 'NSNet', 'DBB', 'ACNet', 'ACNet+']

print("Throughput")
for name, model in zip(names, models):
    print(f"{name}: {get_throughput(model):.0f}")

print("FLOPs")
for name, model in zip(names, models):
    print(f"{name}: {format_human(get_flops(model))}")
