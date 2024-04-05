import torch
from lightning_fabric.utilities import measure_flops
from torch._C._profiler import ProfilerActivity
from torch.autograd.profiler import record_function
from torch.profiler import profile

from src.models import SubResNet, AddResNet


def get_flops(model):
    model.eval()
    model_fwd = lambda: model(x)
    fwd_flops = measure_flops(model, model_fwd)
    return fwd_flops


with torch.device("meta"):
    sub_model = SubResNet('resnet50_SubInceptionV9')
    add_model = AddResNet('resnet50_AddInceptionV1')
    origin_model = AddResNet('resnet50_origin')
    x = torch.randn(1, 3, 224, 224)

sub_flops = get_flops(sub_model)
add_flops = get_flops(add_model)
origin_flops = get_flops(origin_model)

print(f'add/ori {add_flops / origin_flops:.2f}')
print(f'sub/ori {sub_flops / origin_flops:.2f}')
print(f'sub/add {sub_flops / add_flops:.2f}')
#
# with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
#     with record_function("model_inference"):
#         origin_model(x)
# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=1))
#
# with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
#     with record_function("model_inference"):
#         add_model(x)
# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=1))
#
# with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
#     with record_function("model_inference"):
#         sub_model(x)
# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=1))
