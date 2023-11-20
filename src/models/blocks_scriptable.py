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

        x_out = list()
        x = x.permute(4, 0, 1, 2, 3).flatten(0, 1)
        for _, conv in self.blocks.items():
            x_out.append(conv(x))

        x_out = torch.cat(x_out, dim=0).unflatten(0, (n_x * n_conv, n_batch))

        if self.training:
            x_out = x_out[torch.randperm(x_out.size(0))]

        x_out = x_out.unflatten(0, (n_conv, n_x))
        return x_out.sum(1).permute(1, 2, 3, 4, 0)


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
    conv = SubInceptionV7BlockTS(3, 3, 1)
    x = torch.rand(2, 3, 4, 4, 5)

    conv(x)
    # script_conv = torch.jit.script(conv)
    # script_conv(x)
