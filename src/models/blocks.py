import random

import torch
import torch.nn.functional as F
from timm.layers import drop_path
from torch import nn

from src.models.reparam_utils import get_equivalent_kernel_bias, merge_1x1_kxk, fuse_bn, avg_to_kernel, expend_kernel, \
    BNAndPadLayer


def substitute(x, conv_layer, shuffle, neural_drop_rate, training):
    n_x = x.size(-1)
    n_conv = len(conv_layer)
    feature_shape = list(x.size()[1:-1])
    x_out = None

    x = x.permute(4, 0, 1, 2, 3).reshape(-1, *feature_shape)

    for conv in conv_layer:
        out = conv(x)
        out = out.reshape(n_x, -1, *list(out.size()[1:]))
        if x_out is None:
            x_out = out
        else:
            x_out = torch.cat([x_out, out], dim=0)

    if training:
        if shuffle > random.random():
            x_out = x_out[torch.randperm(x_out.size(0))]
        x_out = drop_path(x_out, neural_drop_rate, training)
    x_out = x_out.reshape(n_conv, n_x, *list(x_out.size()[1:]))

    return x_out.sum(1).permute(1, 2, 3, 4, 0)


class BNAndPadLayer(nn.Module):
    def __init__(self,
                 pad_pixels,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True):
        super(BNAndPadLayer, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = self.bn.bias.detach() - self.bn.running_mean * self.bn.weight.detach() / torch.sqrt(
                    self.bn.running_var + self.bn.eps)
            else:
                pad_values = - self.bn.running_mean / torch.sqrt(self.bn.running_var + self.bn.eps)
            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0:self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels:, :] = pad_values
            output[:, :, :, 0:self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels:] = pad_values
        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps


class SubConvBNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_block, stride=1, padding=0, bias=False, stochastic=1.0,
                 bn=nn.BatchNorm2d, neural_drop_rate=0.0, **kwargs):
        super().__init__()
        self.conv_args = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'bias': bias,
            **kwargs
        }
        self.conv_reparam = None
        self.stochastic = stochastic
        self.n_block = n_block
        self.n_flow = 1
        self.neural_drop_rate = neural_drop_rate

        self.blocks = nn.ModuleList()
        for _ in range(n_block):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(**self.conv_args),
                bn(out_channels),
            ))

    def re_parameterization(self):
        self.conv_args['bias'] = True
        self.conv_reparam = nn.Conv2d(**self.conv_args)

        eq_k, eq_b = get_equivalent_kernel_bias(self.blocks, self.n_flow)

        self.conv_reparam.weight.data = eq_k
        self.conv_reparam.bias.data = eq_b

        self.__delattr__('blocks')

    def forward(self, x):
        if self.conv_reparam:
            return self.conv_reparam(x)
        self.n_flow = x.size(-1)
        return substitute(x, self.blocks, self.stochastic, self.neural_drop_rate, self.training)


class NS(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, stochastic=1.0,
                 bn=nn.BatchNorm2d, neural_drop_rate=0.0, groups=1, n_block=0, ratio=1, **kwargs):
        super().__init__()
        self.conv_args = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'bias': bias,
            'groups': groups,
            **kwargs
        }
        hidden_channels1 = int(in_channels * 2)
        hidden_channels2 = int(in_channels * 4)
        self.conv_reparam = None
        self.stochastic = stochastic
        self.n_flow = 1
        self.neural_drop_rate = neural_drop_rate

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
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False, groups=groups, **kwargs),
                BNAndPadLayer(padding, out_channels),
                nn.AvgPool2d(kernel_size=kernel_size, stride=stride),
                bn(out_channels),
            )})
        else:
            self.blocks.update({'1x1': nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (1, 1), bias=False, groups=groups, **kwargs),
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

    def re_parameterization(self):
        self.conv_args['bias'] = True
        self.conv_reparam = nn.Conv2d(**self.conv_args)

        _k0, _b0 = fuse_bn(*self.blocks['kxk'], self.n_flow)

        if self.need_pool:
            _k1, _b1 = fuse_bn(*self.blocks['1x1'][:2], self.n_flow)
            _k11 = avg_to_kernel(self.conv_reparam.out_channels, self.conv_reparam.kernel_size,
                                 self.conv_reparam.groups)
            _k11, _b11 = fuse_bn(_k11.to(self.blocks['1x1'][0].weight.device), self.blocks['1x1'][3], self.n_flow)
            _k1, _b1 = merge_1x1_kxk(_k1, _b1, _k11, _b11, self.conv_reparam.groups)
        else:
            _k1, _b1 = fuse_bn(*self.blocks['1x1'], self.n_flow)
            _k1 = expend_kernel(_k1, self.conv_args['kernel_size'])

        _k3, _b3 = fuse_bn(*self.blocks['dsx2'][:2], self.n_flow)
        _k33, _b33 = fuse_bn(*self.blocks['dsx2'][2:], self.n_flow)
        _k3, _b3 = merge_1x1_kxk(_k3, _b3, _k33, _b33, self.conv_args.get('groups', 1))

        _k4, _b4 = fuse_bn(*self.blocks['dsx4'][:2], self.n_flow)
        _k44, _b44 = fuse_bn(*self.blocks['dsx4'][2:], self.n_flow)
        _k4, _b4 = merge_1x1_kxk(_k4, _b4, _k44, _b44, self.conv_args.get('groups', 1))

        self.conv_reparam.weight.data = sum([_k0, _k1, _k3, _k4])
        self.conv_reparam.bias.data = sum([_b0, _b1, _b3, _b4])
        self.__delattr__('blocks')

    def forward(self, x):
        if self.conv_reparam:
            return self.conv_reparam(x)
        self.n_flow = x.size(-1)
        return substitute(x, self.blocks.values(), self.stochastic, self.neural_drop_rate, self.training)
