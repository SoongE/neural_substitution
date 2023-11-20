import abc
import functools

import torch
from torch import nn

from src.models.blocks import BNAndPadLayer


def implement_jit(class_):
    @functools.wraps(class_, updated=())
    class WrappedClass(class_):
        def __new__(cls, *args, **kwargs):
            instance = class_(*args, **kwargs)
            return torch.jit.script(instance)

    return WrappedClass


class Substitute(abc.ABC):
    def substitute(self, x: torch.Tensor):
        n_batch = x.size(0)
        n_x = x.size(-1)
        n_conv = len(self.blocks)

        x_out = [0] * n_conv
        if self.training:
            rand_idx = torch.randperm(n_conv * n_x) % n_conv
        else:
            rand_idx = torch.arange(n_conv * n_x) % n_conv
        p_idx = 0

        x = x.permute(4, 0, 1, 2, 3).flatten(0, 1)
        for _, conv in self.blocks.items():
            _out = conv(x).unflatten(0, (n_x, n_batch))

            for i, idx in enumerate(rand_idx[p_idx * n_x: (p_idx + 1) * n_x]):
                x_out[idx] = x_out[idx] + _out[i]
            p_idx = p_idx + 1

        x_out = torch.stack(x_out, dim=0)
        return x_out.permute(1, 2, 3, 4, 0)

    def eval_substitution(self, x: torch.Tensor):
        pass


class SubConvBNBlockTS(nn.Module, Substitute):
    def __init__(self, in_channels, out_channels, kernel_size, n_block, stride=1, padding=0, bias=False,
                 bn=nn.BatchNorm2d,
                 groups=1):
        super().__init__()
        self.conv_args = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'bias': bias,
        }
        self.n_block = n_block
        self.n_flow = 1

        self.blocks = nn.ModuleDict()
        for i in range(n_block):
            self.blocks.update({f'kxk_{i}': nn.Sequential(
                nn.Conv2d(**self.conv_args),
                bn(out_channels),
            )})

    def forward(self, x: torch.Tensor):
        self.n_flow = x.size(-1)
        return self.substitute(x)


class SubInceptionV6BlockTS(nn.Module, Substitute):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, bn=nn.BatchNorm2d,
                 groups=1, n_block=0):
        super().__init__()
        self.conv_args = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'bias': bias,
            'groups': groups,
        }
        hidden_channels1 = int(in_channels * 2)
        hidden_channels2 = int(in_channels * 4)
        self.n_flow = 1

        self.blocks = nn.ModuleDict()

        # kxk
        self.blocks.update({'kxk': nn.Sequential(
            nn.Conv2d(**self.conv_args),
            bn(out_channels),
        )})

        self.need_pool = False if stride == 1 else True
        # 1x1
        if self.need_pool:
            self.blocks.update({'1x1': nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False, groups=groups),
                BNAndPadLayer(padding, out_channels),
                nn.AvgPool2d(kernel_size=kernel_size, stride=stride),
                bn(out_channels),
            )})
        else:
            self.blocks.update({'1x1': nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (1, 1), bias=False, groups=groups),
                bn(out_channels),
            )})

        # ds
        self.blocks.update({'dsx2': nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels1, kernel_size=(1, 1), bias=False, groups=groups),
            BNAndPadLayer(padding, hidden_channels1),
            nn.Conv2d(hidden_channels1, out_channels, kernel_size=kernel_size, stride=stride, bias=False,
                      groups=groups),
            bn(out_channels),
        )})

        # ds
        self.blocks.update({'dsx4': nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels2, kernel_size=(1, 1), bias=False, groups=groups),
            BNAndPadLayer(padding, hidden_channels2),
            nn.Conv2d(hidden_channels2, out_channels, kernel_size=kernel_size, stride=stride, bias=False,
                      groups=groups),
            bn(out_channels),
        )})

    def forward(self, x):
        self.n_flow = x.size(-1)
        return self.substitute(x)


class SubInceptionV7BlockTS(nn.Module, Substitute):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, bn=nn.BatchNorm2d,
                 groups=1, n_block=0):
        super().__init__()
        self.conv_args = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'bias': bias,
            'groups': groups,
        }
        hidden_channels1 = int(in_channels * 4)
        hidden_channels2 = int(in_channels * 4)
        self.n_flow = 1

        self.blocks = nn.ModuleDict()

        # kxk
        self.blocks.update({'kxk': nn.Sequential(
            nn.Conv2d(**self.conv_args),
            bn(out_channels),
        )})

        self.need_pool = False if stride == 1 else True
        # 1x1
        if self.need_pool:
            self.blocks.update({'1x1': nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False, groups=groups),
                BNAndPadLayer(padding, out_channels),
                nn.AvgPool2d(kernel_size=kernel_size, stride=stride),
                bn(out_channels),
            )})
        else:
            self.blocks.update({'1x1': nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (1, 1), bias=False, groups=groups),
                bn(out_channels),
            )})

        # ds
        self.blocks.update({'dsx2': nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels1, kernel_size=(1, 1), bias=False, groups=groups),
            BNAndPadLayer(padding, hidden_channels1),
            nn.Conv2d(hidden_channels1, out_channels, kernel_size=kernel_size, stride=stride, bias=False,
                      groups=groups),
            bn(out_channels),
        )})

        # ds
        self.blocks.update({'dsx4': nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels2, kernel_size=(1, 1), bias=False, groups=groups),
            BNAndPadLayer(padding, hidden_channels2),
            nn.Conv2d(hidden_channels2, out_channels, kernel_size=kernel_size, stride=stride, bias=False,
                      groups=groups),
            bn(out_channels),
        )})

    def forward(self, x):
        self.n_flow = x.size(-1)
        return self.substitute(x)


if __name__ == '__main__':
    conv = SubInceptionV6BlockTS(3, 3, 1)
    x = torch.rand(2, 3, 4, 4, 4)

    conv(x)
    # script_conv = torch.jit.script(conv)
    # script_conv(x)
