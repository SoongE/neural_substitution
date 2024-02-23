import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.layers import drop_path

from src.models.utils import ZeroPad1d


def fuse_bn(conv, bn, scale=1):
    kernel = conv.weight if not isinstance(conv, torch.Tensor) else conv
    running_mean = bn.running_mean * scale
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias * scale
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std


def fuse_only_bn(bn, kernel_size, scale=1):
    dim = bn.weight.size(0)
    kernel = torch.zeros((dim, dim, *kernel_size), dtype=bn.weight.dtype, device=bn.weight.device)
    for i in range(dim):
        kernel[i, i % dim, kernel_size[0] // 2, kernel_size[0] // 2] = 1

    running_mean = bn.running_mean * scale
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias * scale
    eps = bn.eps

    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std


def fuse_fc(fc_layers, scale=1):
    w, b = 0, 0
    for fc in fc_layers:
        w += fc.weight.data
        b += fc.bias.data * scale
    return w, b


def expend_kernel(kernel, target_kernel_size):
    if isinstance(target_kernel_size, int):
        target_kernel_size = (target_kernel_size, target_kernel_size)
    H_pixels_to_pad = (target_kernel_size[0] - kernel.size(2)) // 2
    W_pixels_to_pad = (target_kernel_size[1] - kernel.size(3)) // 2
    return F.pad(kernel, [W_pixels_to_pad, W_pixels_to_pad, H_pixels_to_pad, H_pixels_to_pad])


def merge_1x1_kxk(k1, b1, k2, b2, groups=1):
    if groups == 1:
        k = F.conv2d(k2, k1.permute(1, 0, 2, 3))
        b_hat = (k2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3))
    else:
        k_slices = []
        b_slices = []
        k1_T = k1.permute(1, 0, 2, 3)
        k1_group_width = k1.size(0) // groups
        k2_group_width = k2.size(0) // groups
        for g in range(groups):
            k1_T_slice = k1_T[:, g * k1_group_width:(g + 1) * k1_group_width, :, :]
            k2_slice = k2[g * k2_group_width:(g + 1) * k2_group_width, :, :, :]
            k_slices.append(F.conv2d(k2_slice, k1_T_slice))
            b_slices.append(
                (k2_slice * b1[g * k1_group_width:(g + 1) * k1_group_width].reshape(1, -1, 1, 1)).sum((1, 2, 3)))
        k, b_hat = torch.cat(k_slices, dim=0), torch.cat(b_slices)
    return k, b_hat + b2


def avg_to_kernel(channels, kernel_size, groups):
    kernel_size = kernel_size[0]
    input_dim = channels // groups
    k = torch.zeros((channels, input_dim, kernel_size, kernel_size))
    k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size ** 2
    return k


def get_equivalent_kernel_bias(convbn, scale):
    eq_k, eq_b = 0, 0
    for i in range(len(convbn)):
        k, b = fuse_bn(convbn[i][0], convbn[i][1], scale)
        eq_k += k
        eq_b += b
    return eq_k, eq_b


def substitute(x, conv_layer, shuffle, neural_drop_rate, training):
    n_batch = x.size(0)
    n_x = x.size(-1)
    n_conv = len(conv_layer)

    x_out = [0] * n_conv
    if training and shuffle:
        rand_idx = torch.randperm(n_conv * n_x) % n_conv
    else:
        rand_idx = torch.arange(n_conv * n_x) % n_conv
    p_idx = 0

    x = x.permute(4, 0, 1, 2, 3).flatten(0, 1)
    for conv in conv_layer:
        _out = drop_path(conv(x).unflatten(0, (n_x, n_batch)), neural_drop_rate, training)
        # _out = conv(x).unflatten(0, (n_x, n_batch))

        for i, idx in enumerate(rand_idx[p_idx * n_x: (p_idx + 1) * n_x]):
            x_out[idx] = x_out[idx] + _out[i]
        p_idx = p_idx + 1

    x_out = torch.stack(x_out, dim=0)
    return x_out.permute(1, 2, 3, 4, 0)


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


class ConvBNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, n_block=None,
                 stochastic=None, bn=nn.BatchNorm2d, **kwargs):
        super().__init__()
        self.convbn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, **kwargs),
            bn(out_channels),
        )

    def forward(self, x):
        return self.convbn(x)


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


class SubInceptionV1in1Block(nn.Module):
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
            self.blocks.append(AddInceptionV1Block(**self.conv_args))

    def re_parameterization(self):
        self.conv_args['bias'] = True
        self.conv_reparam = nn.Conv2d(**self.conv_args)

        weight, bias = 0, 0

        for block in self.blocks:
            block.re_parameterization()
            weight = weight + block.conv_reparam.weight.data
            bias = bias + block.conv_reparam.bias.data

        self.conv_reparam.weight.data = weight
        self.conv_reparam.bias.data = bias * self.n_flow

        self.__delattr__('blocks')

    def forward(self, x):
        if self.conv_reparam:
            return self.conv_reparam(x)
        self.n_flow = x.size(-1)
        return substitute(x, self.blocks, self.stochastic, self.neural_drop_rate, self.training)


class SubInceptionV1Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, stochastic=1.0,
                 bn=nn.BatchNorm2d, neural_drop_rate=0.0, hidden_channels=None, n_block=0, **kwargs):
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
        hidden_channels = hidden_channels if hidden_channels else in_channels
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

        # 1x1
        self.blocks.update({'1x1': nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride, bias=False, **kwargs),
            bn(out_channels),
        )})

        # 1x1-kxk
        self.blocks.update({'1x1-kxk': nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=(1, 1), bias=False, groups=kwargs.get('groups', 1)),
            BNAndPadLayer(padding, hidden_channels),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False, **kwargs),
            bn(out_channels),
        )})

        # 1x1-avg
        self.blocks.update({'1x1-avg': nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False, **kwargs),
            BNAndPadLayer(padding, out_channels),
            nn.AvgPool2d(kernel_size=kernel_size, stride=stride),
            bn(out_channels),
        )})

    def re_parameterization(self):
        self.conv_args['bias'] = True
        self.conv_reparam = nn.Conv2d(**self.conv_args)

        _k0, _b0 = fuse_bn(*self.blocks['kxk'], self.n_flow)

        _k1, _b1 = fuse_bn(*self.blocks['1x1'], self.n_flow)
        _k1 = expend_kernel(_k1, self.conv_args['kernel_size'])

        _k2, _b2 = fuse_bn(*self.blocks['1x1-kxk'][:2], self.n_flow)
        _k22, _b22 = fuse_bn(*self.blocks['1x1-kxk'][2:], self.n_flow)
        _k2, _b2 = merge_1x1_kxk(_k2, _b2, _k22, _b22, self.conv_reparam.groups)

        _k3, _b3 = fuse_bn(*self.blocks['1x1-avg'][:2], self.n_flow)
        _k33 = avg_to_kernel(self.conv_reparam.out_channels, self.conv_reparam.kernel_size, self.conv_reparam.groups)
        _k33, _b33 = fuse_bn(_k33.to(self.blocks['1x1-avg'][0].weight.device), self.blocks['1x1-avg'][3], self.n_flow)
        _k3, _b3 = merge_1x1_kxk(_k3, _b3, _k33, _b33, self.conv_reparam.groups)

        self.conv_reparam.weight.data = sum([_k0, _k1, _k2, _k3])
        self.conv_reparam.bias.data = sum([_b0, _b1, _b2, _b3])
        self.__delattr__('blocks')

    def forward(self, x):
        if self.conv_reparam:
            return self.conv_reparam(x)
        self.n_flow = x.size(-1)
        return substitute(x, self.blocks.values(), self.stochastic, self.neural_drop_rate, self.training)


class SubInceptionV2Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, stochastic=1.0,
                 bn=nn.BatchNorm2d, neural_drop_rate=0.0, n_block=0, hidden_channels=None, **kwargs):
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
        self.n_flow = 1
        self.neural_drop_rate = neural_drop_rate

        self.blocks = nn.ModuleDict()

        # kxk
        self.blocks.update({'kxk': nn.Sequential(
            nn.Conv2d(**self.conv_args),
            bn(out_channels),
        )})

        _padding = (1 // 2, 3 // 2)
        # 1x3
        self.blocks.update({'1x3': nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), bias=False, stride=stride, padding=_padding,
                      **kwargs),
            bn(out_channels),
        )})

        _padding = (3 // 2, 1 // 2)
        # 3x1
        self.blocks.update({'3x1': nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), bias=False, stride=stride, padding=_padding,
                      **kwargs),
            bn(out_channels),
        )})

    def re_parameterization(self):
        self.conv_args['bias'] = True
        self.conv_reparam = nn.Conv2d(**self.conv_args)

        _k0, _b0 = fuse_bn(*self.blocks['kxk'], self.n_flow)

        _k1, _b1 = fuse_bn(*self.blocks['1x3'], self.n_flow)
        _k1 = expend_kernel(_k1, self.conv_args['kernel_size'])

        _k2, _b2 = fuse_bn(*self.blocks['3x1'], self.n_flow)
        _k2 = expend_kernel(_k2, self.conv_args['kernel_size'])

        self.conv_reparam.weight.data = sum([_k0, _k1, _k2])
        self.conv_reparam.bias.data = sum([_b0, _b1, _b2])
        self.__delattr__('blocks')

    def forward(self, x):
        if self.conv_reparam:
            return self.conv_reparam(x)
        self.n_flow = x.size(-1)
        return substitute(x, self.blocks.values(), self.stochastic, self.neural_drop_rate, self.training)


class SubInceptionV3Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, stochastic=1.0,
                 bn=nn.BatchNorm2d, neural_drop_rate=0.0, n_block=0, hidden_channels=None, **kwargs):
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
        self.n_flow = 1
        self.neural_drop_rate = neural_drop_rate

        self.blocks = nn.ModuleDict()

        # kxk
        self.blocks.update({'kxk': nn.Sequential(
            nn.Conv2d(**self.conv_args),
            bn(out_channels),
        )})

        # 1x1
        self.blocks.update({'1x1': nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride, bias=False, **kwargs),
            bn(out_channels),
        )})

        _padding = (1 // 2, 3 // 2)
        # 1x3
        self.blocks.update({'1x3': nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), bias=False, stride=stride, padding=_padding,
                      **kwargs),
            bn(out_channels),
        )})

        _padding = (3 // 2, 1 // 2)
        # 3x1
        self.blocks.update({'3x1': nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), bias=False, stride=stride, padding=_padding,
                      **kwargs),
            bn(out_channels),
        )})

    def re_parameterization(self):
        self.conv_args['bias'] = True
        self.conv_reparam = nn.Conv2d(**self.conv_args)

        _k0, _b0 = fuse_bn(*self.blocks['kxk'], self.n_flow)

        _k1, _b1 = fuse_bn(*self.blocks['1x1'], self.n_flow)
        _k1 = expend_kernel(_k1, self.conv_args['kernel_size'])

        _k2, _b2 = fuse_bn(*self.blocks['1x3'], self.n_flow)
        _k2 = expend_kernel(_k2, self.conv_args['kernel_size'])

        _k3, _b3 = fuse_bn(*self.blocks['3x1'], self.n_flow)
        _k3 = expend_kernel(_k3, self.conv_args['kernel_size'])

        self.conv_reparam.weight.data = sum([_k0, _k1, _k2, _k3])
        self.conv_reparam.bias.data = sum([_b0, _b1, _b2, _b3])
        self.__delattr__('blocks')

    def forward(self, x):
        if self.conv_reparam:
            return self.conv_reparam(x)
        self.n_flow = x.size(-1)
        return substitute(x, self.blocks.values(), self.stochastic, self.neural_drop_rate, self.training)


class SubInceptionV4Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, stochastic=1.0,
                 bn=nn.BatchNorm2d, neural_drop_rate=0.0, n_block=0, ratio=1, **kwargs):
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
        hidden_channels = int(in_channels * ratio)
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

        self.need_pool = False if (stride == 1 and padding == 1) else True
        # 1x1
        if self.need_pool:
            self.blocks.update({'1x1': nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False, **kwargs),
                BNAndPadLayer(padding, out_channels),
                nn.AvgPool2d(kernel_size=kernel_size, stride=stride),
                bn(out_channels),
            )})
        else:
            self.blocks.update({'1x1': nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (1, 1), stride=stride, bias=False, **kwargs),
                bn(out_channels),
            )})

        # ds
        self.blocks.update({'ds': nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=(1, 1), bias=False, groups=kwargs.get('groups', 1)),
            BNAndPadLayer(padding, hidden_channels),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False, **kwargs),
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

        _k3, _b3 = fuse_bn(*self.blocks['ds'][:2], self.n_flow)
        _k33, _b33 = fuse_bn(*self.blocks['ds'][2:], self.n_flow)
        _k3, _b3 = merge_1x1_kxk(_k3, _b3, _k33, _b33)

        self.conv_reparam.weight.data = sum([_k0, _k1, _k3])
        self.conv_reparam.bias.data = sum([_b0, _b1, _b3])
        self.__delattr__('blocks')

    def forward(self, x):
        if self.conv_reparam:
            return self.conv_reparam(x)
        self.n_flow = x.size(-1)
        return substitute(x, self.blocks.values(), self.stochastic, self.neural_drop_rate, self.training)


class SubInceptionV5Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, stochastic=1.0,
                 bn=nn.BatchNorm2d, neural_drop_rate=0.0, n_block=0, ratio=1, **kwargs):
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
        hidden_channels = int(in_channels * ratio)
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
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False, **kwargs),
                BNAndPadLayer(padding, out_channels),
                nn.AvgPool2d(kernel_size=kernel_size, stride=stride),
                bn(out_channels),
            )})
        else:
            self.blocks.update({'1x1': nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (1, 1), bias=False, **kwargs),
                bn(out_channels),
            )})

        # ds
        self.blocks.update({'ds': nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=(1, 1), bias=False, groups=kwargs.get('groups', 1)),
            BNAndPadLayer(padding, hidden_channels),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False, **kwargs),
            bn(out_channels),
        )})

        _padding = (1 // 2, 3 // 2)
        # 1x3
        self.blocks.update({'1x3': nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), bias=False, stride=stride, padding=_padding,
                      **kwargs),
            bn(out_channels),
        )})

        _padding = (3 // 2, 1 // 2)
        # 3x1
        self.blocks.update({'3x1': nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), bias=False, stride=stride, padding=_padding,
                      **kwargs),
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

        _k3, _b3 = fuse_bn(*self.blocks['ds'][:2], self.n_flow)
        _k33, _b33 = fuse_bn(*self.blocks['ds'][2:], self.n_flow)
        _k3, _b3 = merge_1x1_kxk(_k3, _b3, _k33, _b33)

        _k4, _b4 = fuse_bn(*self.blocks['1x3'], self.n_flow)
        _k4 = expend_kernel(_k4, self.conv_args['kernel_size'])

        _k5, _b5 = fuse_bn(*self.blocks['3x1'], self.n_flow)
        _k5 = expend_kernel(_k5, self.conv_args['kernel_size'])

        self.conv_reparam.weight.data = sum([_k0, _k1, _k3, _k4, _k5])
        self.conv_reparam.bias.data = sum([_b0, _b1, _b3, _b4, _b5])
        self.__delattr__('blocks')

    def forward(self, x):
        if self.conv_reparam:
            return self.conv_reparam(x)
        self.n_flow = x.size(-1)
        return substitute(x, self.blocks.values(), self.stochastic, self.neural_drop_rate, self.training)


class SubInceptionV6Block(nn.Module):
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


class SubInceptionV7Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, stochastic=False,
                 bn=nn.BatchNorm2d, neural_drop_rate=0.0, n_block=0, ratio=1, **kwargs):
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
        hidden_channels1 = int(in_channels * 2)
        hidden_channels2 = int(in_channels * 2)
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

        self.blocks.update({'1x1': nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride, bias=False, **kwargs),
            bn(out_channels),
        )})

        # self.need_pool = False if stride == 1 else True
        # if self.need_pool:
        #     self.blocks.update({'1x12': nn.Sequential(
        #         nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False, **kwargs),
        #         BNAndPadLayer(padding, out_channels),
        #         nn.AvgPool2d(kernel_size=kernel_size, stride=stride),
        #         bn(out_channels),
        #     )})
        # else:
        #     self.blocks.update({'1x12': nn.Identity()})

        # ds
        self.blocks.update({'dsx2': nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels2, kernel_size=(1, 1), bias=False, groups=kwargs.get('groups', 1)),
            BNAndPadLayer(padding, hidden_channels2),
            nn.Conv2d(hidden_channels2, out_channels, kernel_size=kernel_size, stride=stride, bias=False,
                      groups=kwargs.get('groups', 1)),
            bn(out_channels),
        )})

    def re_parameterization(self):
        self.conv_args['bias'] = True
        self.conv_reparam = nn.Conv2d(**self.conv_args)

        _k0, _b0 = fuse_bn(*self.blocks['kxk'], self.n_flow)

        _k1, _b1 = fuse_bn(*self.blocks['1x1'], self.n_flow)
        _k1 = expend_kernel(_k1, self.conv_args['kernel_size'])

        _k3, _b3 = fuse_bn(*self.blocks['dsx2'][:2], self.n_flow)
        _k33, _b33 = fuse_bn(*self.blocks['dsx2'][2:], self.n_flow)
        _k3, _b3 = merge_1x1_kxk(_k3, _b3, _k33, _b33, self.conv_args.get('groups', 1))

        # _k4, _b4 = fuse_bn(*self.blocks['dsx4'][:2], self.n_flow)
        # _k44, _b44 = fuse_bn(*self.blocks['dsx4'][2:], self.n_flow)
        # _k4, _b4 = merge_1x1_kxk(_k4, _b4, _k44, _b44, self.conv_args.get('groups', 1))

        self.conv_reparam.weight.data = sum([_k0, _k3, _k1])
        self.conv_reparam.bias.data = sum([_b0, _b3, _b1])
        self.__delattr__('blocks')

    def forward(self, x):
        if self.conv_reparam:
            return self.conv_reparam(x)
        self.n_flow = x.size(-1)
        return substitute(x, self.blocks.values(), self.stochastic, self.neural_drop_rate, self.training)


class SubInceptionV8Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, stochastic=False,
                 bn=nn.BatchNorm2d, neural_drop_rate=0.0, n_block=0, ratio=1, **kwargs):
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
        hidden_channels1 = int(in_channels * 2)
        hidden_channels2 = int(in_channels * 2)
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

        self.blocks.update({'1x1': nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride, bias=False, **kwargs),
            bn(out_channels),
        )})

        self.need_pool = False if stride == 1 else True
        # 1x1
        if self.need_pool:
            self.blocks.update({'1x12': nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False, **kwargs),
                BNAndPadLayer(padding, out_channels),
                nn.AvgPool2d(kernel_size=kernel_size, stride=stride),
                bn(out_channels),
            )})
        else:
            self.blocks.update({'1x12': bn(out_channels)})

        # ds
        self.blocks.update({'dsx2': nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels2, kernel_size=(1, 1), bias=False, groups=kwargs.get('groups', 1)),
            BNAndPadLayer(padding, hidden_channels2),
            nn.Conv2d(hidden_channels2, out_channels, kernel_size=kernel_size, stride=stride, bias=False,
                      groups=kwargs.get('groups', 1)),
            bn(out_channels),
        )})

    def re_parameterization(self):
        self.conv_args['bias'] = True
        self.conv_reparam = nn.Conv2d(**self.conv_args)

        _k0, _b0 = fuse_bn(*self.blocks['kxk'], self.n_flow)

        _k1, _b1 = fuse_bn(*self.blocks['1x1'], self.n_flow)
        _k1 = expend_kernel(_k1, self.conv_args['kernel_size'])

        _k3, _b3 = fuse_bn(*self.blocks['dsx2'][:2], self.n_flow)
        _k33, _b33 = fuse_bn(*self.blocks['dsx2'][2:], self.n_flow)
        _k3, _b3 = merge_1x1_kxk(_k3, _b3, _k33, _b33, self.conv_args.get('groups', 1))

        # _k4, _b4 = fuse_bn(*self.blocks['dsx4'][:2], self.n_flow)
        # _k44, _b44 = fuse_bn(*self.blocks['dsx4'][2:], self.n_flow)
        # _k4, _b4 = merge_1x1_kxk(_k4, _b4, _k44, _b44, self.conv_args.get('groups', 1))

        self.conv_reparam.weight.data = sum([_k0, _k3, _k1])
        self.conv_reparam.bias.data = sum([_b0, _b3, _b1])
        self.__delattr__('blocks')

    def forward(self, x):
        if self.conv_reparam:
            return self.conv_reparam(x)
        self.n_flow = x.size(-1)
        return substitute(x, self.blocks.values(), self.stochastic, self.neural_drop_rate, self.training)


class SubInceptionV9Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, stochastic=False,
                 bn=nn.BatchNorm2d, neural_drop_rate=0.0, n_block=0, ratio=1, **kwargs):
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
        hidden_channels1 = int(in_channels * 2)
        hidden_channels2 = int(in_channels * 2)
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

        self.blocks.update({'1x1': nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride, bias=False, **kwargs),
            bn(out_channels),
        )})

        # ds
        self.blocks.update({'dsx1': nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels1, kernel_size=(1, 1), bias=False, groups=kwargs.get('groups', 1)),
            BNAndPadLayer(padding, hidden_channels1),
            nn.Conv2d(hidden_channels1, out_channels, kernel_size=kernel_size, stride=stride, bias=False,
                      groups=kwargs.get('groups', 1)),
            bn(out_channels),
        )})

        # ds
        self.blocks.update({'dsx2': nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels2, kernel_size=(1, 1), bias=False, groups=kwargs.get('groups', 1)),
            BNAndPadLayer(padding, hidden_channels2),
            nn.Conv2d(hidden_channels2, out_channels, kernel_size=kernel_size, stride=stride, bias=False,
                      groups=kwargs.get('groups', 1)),
            bn(out_channels),
        )})

    def re_parameterization(self):
        self.conv_args['bias'] = True
        self.conv_reparam = nn.Conv2d(**self.conv_args)

        _k0, _b0 = fuse_bn(*self.blocks['kxk'], self.n_flow)

        _k1, _b1 = fuse_bn(*self.blocks['1x1'], self.n_flow)
        _k1 = expend_kernel(_k1, self.conv_args['kernel_size'])

        _k4, _b4 = fuse_bn(*self.blocks['dsx1'][:2], self.n_flow)
        _k44, _b44 = fuse_bn(*self.blocks['dsx1'][2:], self.n_flow)
        _k4, _b4 = merge_1x1_kxk(_k4, _b4, _k44, _b44, self.conv_args.get('groups', 1))

        _k3, _b3 = fuse_bn(*self.blocks['dsx2'][:2], self.n_flow)
        _k33, _b33 = fuse_bn(*self.blocks['dsx2'][2:], self.n_flow)
        _k3, _b3 = merge_1x1_kxk(_k3, _b3, _k33, _b33, self.conv_args.get('groups', 1))

        self.conv_reparam.weight.data = sum([_k0, _k3, _k4, _k1])
        self.conv_reparam.bias.data = sum([_b0, _b3, _b4, _b1])
        self.__delattr__('blocks')

    def forward(self, x):
        if self.conv_reparam:
            return self.conv_reparam(x)
        self.n_flow = x.size(-1)
        return substitute(x, self.blocks.values(), self.stochastic, self.neural_drop_rate, self.training)


class AddConvBNBlock(SubConvBNBlock):
    def forward(self, x):
        if self.conv_reparam:
            return self.conv_reparam(x)

        out = 0
        for block in self.blocks:
            out = out + block(x)

        return out


class AddInceptionV1in1Block(SubInceptionV1in1Block):
    def forward(self, x):
        if self.conv_reparam:
            return self.conv_reparam(x)

        out = 0
        for block in self.blocks:
            out = out + block(x)

        return out


class AddInceptionV1Block(SubInceptionV1Block):
    def forward(self, x):
        if self.conv_reparam:
            return self.conv_reparam(x)

        out = 0
        for name, block in self.blocks.items():
            out = out + block(x)

        return out


class AddInceptionV2Block(SubInceptionV2Block):
    def forward(self, x):
        if self.conv_reparam:
            return self.conv_reparam(x)

        out = 0
        for name, block in self.blocks.items():
            out = out + block(x)

        return out


class AddInceptionV3Block(SubInceptionV3Block):
    def forward(self, x):
        if self.conv_reparam:
            return self.conv_reparam(x)

        out = 0
        for name, block in self.blocks.items():
            out = out + block(x)

        return out


class SubLinear(nn.Module):
    def __init__(self, in_features, out_features, FA):
        super().__init__()
        self.linear_args = {
            'in_features': in_features,
            'out_features': out_features,
            'bias': True,
        }
        self.linear_reparam = None
        self.FA = FA
        self.n_flow = 1

        self.layers = nn.ModuleList()
        for _ in range(FA):
            self.layers.append(nn.Linear(**self.linear_args))

    def re_parameterization(self):
        self.linear_reparam = nn.Linear(**self.linear_args)
        w, b = fuse_fc(self.layers, self.n_flow)
        self.linear_reparam.weight.data = w
        self.linear_reparam.bias.data = b

        self.__delattr__('layers')

    def forward(self, x, n_x=None):
        if self.linear_reparam:
            return self.linear_reparam(x)
        self.n_flow = n_x
        x_out = None
        for fc in self.layers:
            out = fc(x)
            if x_out is None:
                x_out = out
            else:
                x_out = x_out + out
        return x_out.reshape(n_x, -1, x_out.size(-1)).sum(0)


class SubMlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            n_blocks=4,
            N=14,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features or in_features
        self.hidden_features = hidden_features or in_features
        self.n_blocks = n_blocks
        self.N = N

        self.fc1 = None
        self.fc2 = None

        self.fc_list1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_features, self.hidden_features, (1, 1), bias=False),
                nn.BatchNorm2d(self.hidden_features),
            ) for _ in range(n_blocks)])
        self.act = nn.ReLU()
        self.fc_list2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.hidden_features, self.out_features, (1, 1), bias=False),
                nn.BatchNorm2d(self.out_features),
            ) for _ in range(n_blocks)])
        self.drop = nn.Dropout2d(0.2, inplace=True)

    def forward(self, x):
        x = rearrange(x, 'b (n1 n2) c -> b c n1 n2', n1=self.N, n2=self.N)
        if self.fc1:
            out = self.fc2(self.act(self.fc1(x)))
        else:
            xs = self.substitute(x.unsqueeze(-1), self.fc_list1, 0, 0, self.training)
            xs = self.guided_activation(xs)
            xs = self.substitute(xs, self.fc_list2, 0, 0, self.training)
            out = xs.sum(-1)
        return rearrange(out, 'b c n1 n2 -> b (n1 n2) c')

    def substitute(self, x, fc_layer, shuffle, neural_drop_rate, training):
        n_x = x.size(-1)
        n_conv = len(fc_layer)
        feature_shape = list(x.size()[1:-1])
        x_out = list()

        x = x.permute(4, 0, 1, 2, 3).reshape(-1, *feature_shape)

        for fc in fc_layer:
            x_out.append(self.drop(fc(x)))

        x_out = torch.cat(x_out, dim=0)
        x_out = x_out.reshape(n_x * n_conv, -1, *list(x_out.size()[1:]))

        if training:
            # if shuffle > random.random():
            x_out = x_out[torch.randperm(x_out.size(0))]
            # x_out = drop_path(x_out, neural_drop_rate, training)
        x_out = x_out.reshape(n_conv, n_x, *list(x_out.size()[1:]))

        return x_out.sum(1).permute(1, 2, 3, 4, 0)

    def guided_activation(self, xs):
        x = torch.sum(xs, dim=4).squeeze(-1)
        x = self.act(x)

        dead_idx = x == 0
        xs[dead_idx] = 0
        return xs

    def fc_list_re_parameterization(self, in_features, out_features, fc_list, scale):
        fc = nn.Conv2d(in_features, out_features, (1, 1))
        eq_k, eq_b = get_equivalent_kernel_bias(fc_list, scale)
        fc.weight.data = eq_k
        fc.bias.data = eq_b

        return fc

    def re_parameterization(self):
        self.fc1 = self.fc_list_re_parameterization(self.in_features, self.hidden_features, self.fc_list1, 1)
        self.fc2 = self.fc_list_re_parameterization(self.hidden_features, self.out_features, self.fc_list2,
                                                    self.n_blocks)

        self.__delattr__('fc_list1')
        self.__delattr__('fc_list2')


class SubMlpV2(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            n_blocks=4,
            N=14,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features or in_features
        # self.hidden_features = hidden_features or in_features
        self.hidden_features1 = int(in_features * 2)
        self.hidden_features2 = int(in_features * 4)
        self.n_blocks = n_blocks
        self.N = N

        self.fc1 = None
        self.fc2 = None
        self.fc3 = None
        self.fc4 = None

        self.fc_list1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_features, self.hidden_features1, (1, 1), bias=False),
                nn.BatchNorm2d(self.hidden_features1),
            ) for _ in range(n_blocks)])
        self.fc_list2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.hidden_features1, self.hidden_features2, (1, 1), bias=False),
                nn.BatchNorm2d(self.hidden_features2),
            ) for _ in range(n_blocks)])
        self.act = nn.ReLU()
        self.fc_list3 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.hidden_features2, self.hidden_features1, (1, 1), bias=False),
                nn.BatchNorm2d(self.hidden_features1),
            ) for _ in range(n_blocks)])
        self.fc_list4 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.hidden_features1, self.out_features, (1, 1), bias=False),
                nn.BatchNorm2d(self.out_features),
            ) for _ in range(n_blocks)])

        self.drop = nn.Dropout2d(0.2, inplace=True)

    def forward(self, x):
        x = rearrange(x, 'b (n1 n2) c -> b c n1 n2', n1=self.N, n2=self.N)
        if self.fc1:
            x = self.act(self.fc1(x))
            x = self.act(self.fc2(x))
            x = self.act(self.fc3(x))
            out = self.fc4(x)
        else:
            xs = self.substitute(x.unsqueeze(-1), self.fc_list1, 0, 0, self.training)
            xs = self.guided_activation(xs)
            xs = self.substitute(xs, self.fc_list2, 0, 0, self.training)
            xs = self.guided_activation(xs)
            xs = self.substitute(xs, self.fc_list3, 0, 0, self.training)
            xs = self.guided_activation(xs)
            xs = self.substitute(xs, self.fc_list4, 0, 0, self.training)
            out = xs.sum(-1)
        return rearrange(out, 'b c n1 n2 -> b (n1 n2) c')

    def substitute(self, x, fc_layer, shuffle, neural_drop_rate, training):
        n_x = x.size(-1)
        n_conv = len(fc_layer)
        feature_shape = list(x.size()[1:-1])
        x_out = list()

        x = x.permute(4, 0, 1, 2, 3).reshape(-1, *feature_shape)

        for fc in fc_layer:
            x_out.append(fc(x))

        x_out = torch.cat(x_out, dim=0)
        x_out = x_out.reshape(n_x * n_conv, -1, *list(x_out.size()[1:]))

        if training:
            x_out = x_out[torch.randperm(x_out.size(0))]
        x_out = x_out.reshape(n_conv, n_x, *list(x_out.size()[1:]))

        return x_out.sum(1).permute(1, 2, 3, 4, 0)

    def guided_activation(self, xs):
        x = torch.sum(xs, dim=4).squeeze(-1)
        x = self.act(x)

        dead_idx = x == 0
        xs[dead_idx] = 0
        return xs

    def fc_list_re_parameterization(self, in_features, out_features, fc_list, scale):
        fc = nn.Conv2d(in_features, out_features, (1, 1))
        eq_k, eq_b = get_equivalent_kernel_bias(fc_list, scale)
        fc.weight.data = eq_k
        fc.bias.data = eq_b

        return fc

    def re_parameterization(self):
        self.fc1 = self.fc_list_re_parameterization(self.in_features, self.hidden_features1, self.fc_list1, 1)
        self.fc2 = self.fc_list_re_parameterization(self.hidden_features1, self.hidden_features2, self.fc_list2,
                                                    self.n_blocks)
        self.fc3 = self.fc_list_re_parameterization(self.hidden_features2, self.hidden_features1, self.fc_list3,
                                                    self.n_blocks)
        self.fc4 = self.fc_list_re_parameterization(self.hidden_features1, self.out_features, self.fc_list4,
                                                    self.n_blocks)

        self.__delattr__('fc_list1')
        self.__delattr__('fc_list2')
        self.__delattr__('fc_list3')
        self.__delattr__('fc_list4')


class SubMlpV3(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            n_blocks=3,
            N=14,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features or in_features
        self.hidden_features = int(in_features * 4)
        self.n_blocks = n_blocks
        self.N = N

        self.fc1 = None
        self.fc2 = None

        self.fc_list1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_features, int(self.in_features * (i + 2)), (1, 1), bias=False),
                nn.BatchNorm2d(int(self.in_features * (i + 2))),
                Rearrange('b c h w -> b h w c'),
                ZeroPad1d((self.hidden_features - self.in_features * (i + 2)) // 2),
                Rearrange('b h w c -> b c h w'),
            ) for i in range(n_blocks)])
        self.act = nn.ReLU()
        self.fc_list2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.hidden_features, self.out_features, (1, 1), bias=False),
                nn.BatchNorm2d(self.out_features),
            ) for _ in range(n_blocks)])

    def forward(self, x):
        x = rearrange(x, 'b (n1 n2) c -> b c n1 n2', n1=self.N, n2=self.N)
        if self.fc1:
            out = self.fc2(self.act(self.fc1(x)))
        else:
            xs = self.substitute(x.unsqueeze(-1), self.fc_list1, 0, 0, self.training)
            xs = self.guided_activation(xs)
            xs = self.substitute(xs, self.fc_list2, 0, 0, self.training)
            out = xs.sum(-1)
        return rearrange(out, 'b c n1 n2 -> b (n1 n2) c')

    def substitute(self, x, fc_layer, shuffle, neural_drop_rate, training):
        n_x = x.size(-1)
        n_conv = len(fc_layer)
        feature_shape = list(x.size()[1:-1])
        x_out = list()

        x = x.permute(4, 0, 1, 2, 3).reshape(-1, *feature_shape)

        for fc in fc_layer:
            x_out.append(fc(x))

        x_out = torch.cat(x_out, dim=0)
        x_out = x_out.reshape(n_x * n_conv, -1, *list(x_out.size()[1:]))

        if training:
            # if shuffle > random.random():
            x_out = x_out[torch.randperm(x_out.size(0))]
            # x_out = drop_path(x_out, neural_drop_rate, training)
        x_out = x_out.reshape(n_conv, n_x, *list(x_out.size()[1:]))

        return x_out.sum(1).permute(1, 2, 3, 4, 0)

    def guided_activation(self, xs):
        x = torch.sum(xs, dim=4).squeeze(-1)
        x = self.act(x)

        dead_idx = x == 0
        xs[dead_idx] = 0
        return xs

    def fc_list_re_parameterization(self, in_features, out_features, fc_list, scale, pad):
        fc = nn.Conv2d(in_features, out_features, (1, 1))
        eq_k, eq_b = self.get_equivalent_kernel_bias(fc_list, scale, pad)
        fc.weight.data = eq_k
        fc.bias.data = eq_b

        return fc

    def get_equivalent_kernel_bias(self, convbn, scale, pad=False):
        eq_k, eq_b = 0, 0
        for i in range(len(convbn)):
            k, b = fuse_bn(convbn[i][0], convbn[i][1], scale)
            if pad:
                pad = (self.hidden_features - k.size(0)) // 2
                k = nn.functional.pad(k.permute(1, 2, 3, 0), (pad, pad), value=0).permute(3, 0, 1, 2)
                b = nn.functional.pad(b, (pad, pad), value=0)
            eq_k += k
            eq_b += b
        return eq_k, eq_b

    def re_parameterization(self):
        self.fc1 = self.fc_list_re_parameterization(self.in_features, self.hidden_features, self.fc_list1, 1, True)
        self.fc2 = self.fc_list_re_parameterization(self.hidden_features, self.out_features, self.fc_list2,
                                                    self.n_blocks, False)

        self.__delattr__('fc_list1')
        self.__delattr__('fc_list2')


class SubMlpV4(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            n_blocks=4,
            N=14,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features or in_features
        self.hidden_features = int(in_features * 4)
        self.n_blocks = n_blocks
        self.N = N
        self.conv_args = {
            'kernel_size': (3, 3),
            'stride': 1,
            'padding': 1,
            'bias': False,
        }

        self.conv1 = None
        self.conv2 = None
        self.fc1 = None
        self.fc2 = None

        self.act = nn.ReLU(inplace=True)

        self.conv_list1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_features, self.in_features, groups=self.in_features, **self.conv_args),
                nn.BatchNorm2d(self.in_features),
            ) for _ in range(n_blocks)])
        self.fc_list1 = nn.ModuleList([nn.Conv2d(self.in_features, self.hidden_features, (1, 1), bias=True)
                                       for _ in range(n_blocks)])

        self.conv_list2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.hidden_features, self.hidden_features, groups=self.hidden_features, **self.conv_args),
                nn.BatchNorm2d(self.hidden_features),
            ) for _ in range(n_blocks)])
        self.fc_list2 = nn.ModuleList([nn.Conv2d(self.hidden_features, self.out_features, (1, 1), bias=True)
                                       for _ in range(n_blocks)])

    def forward(self, x):
        x = rearrange(x, 'b (n1 n2) c -> b c n1 n2', n1=self.N, n2=self.N)
        if self.conv1:
            x = self.act(self.conv1(x))
            x = self.act(self.fc1(x))
            x = self.act(self.conv2(x))
            out = self.fc2(x)
        else:
            xs = self.substitute(x.unsqueeze(-1), self.conv_list1, 0, 0, self.training)
            xs = self.guided_activation(xs)
            xs = self.substitute(xs, self.fc_list1, 0, 0, self.training)
            xs = self.guided_activation(xs)
            xs = self.substitute(xs, self.conv_list2, 0, 0, self.training)
            xs = self.guided_activation(xs)
            xs = self.substitute(xs, self.fc_list2, 0, 0, self.training)
            out = xs.sum(-1)
        return rearrange(out, 'b c n1 n2 -> b (n1 n2) c')

    def substitute(self, x, fc_layer, shuffle, neural_drop_rate, training):
        n_x = x.size(-1)
        n_conv = len(fc_layer)
        feature_shape = list(x.size()[1:-1])
        x_out = list()

        x = x.permute(4, 0, 1, 2, 3).reshape(-1, *feature_shape)

        for fc in fc_layer:
            x_out.append(fc(x))

        x_out = torch.cat(x_out, dim=0)
        x_out = x_out.reshape(n_x * n_conv, -1, *list(x_out.size()[1:]))

        if training:
            x_out = x_out[torch.randperm(x_out.size(0))]
        x_out = x_out.reshape(n_conv, n_x, *list(x_out.size()[1:]))

        return x_out.sum(1).permute(1, 2, 3, 4, 0)

    def guided_activation(self, xs):
        x = torch.sum(xs, dim=4).squeeze(-1)
        x = self.act(x)

        dead_idx = x == 0
        xs[dead_idx] = 0
        return xs

    def re_parameterization_list(self, blocks, in_features, out_features, scale, **conv_args):
        conv_args['bias'] = True
        groups = in_features if in_features == out_features else 1
        conv = nn.Conv2d(in_features, out_features, groups=groups, **conv_args)
        if conv.kernel_size == (3, 3):
            eq_k, eq_b = get_equivalent_kernel_bias(blocks, scale)
        else:
            eq_k, eq_b = fuse_fc(blocks, scale)

        conv.weight.data = eq_k
        conv.bias.data = eq_b

        return conv

    def re_parameterization(self):
        self.conv1 = self.re_parameterization_list(self.conv_list1, self.in_features, self.in_features, 1,
                                                   **self.conv_args)
        self.fc1 = self.re_parameterization_list(self.fc_list1, self.in_features, self.hidden_features, self.n_blocks,
                                                 kernel_size=(1, 1), bias=True)

        self.conv2 = self.re_parameterization_list(self.conv_list2, self.hidden_features, self.hidden_features,
                                                   self.n_blocks, **self.conv_args)
        self.fc2 = self.re_parameterization_list(self.fc_list2, self.in_features, self.hidden_features, self.n_blocks,
                                                 kernel_size=(1, 1), bias=True)

        self.__delattr__('conv_list1')
        self.__delattr__('fc_list1')
        self.__delattr__('conv_list2')
        self.__delattr__('fc_list2')


class SubMlpV5(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            n_blocks=3,
            N=14,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features or in_features
        self.hidden_features = int(in_features * 4)
        self.n_blocks = n_blocks
        self.N = N
        self.conv_args = {
            'kernel_size': (3, 3),
            'stride': 1,
            'padding': 1,
            'bias': False,
        }

        self.conv1 = None
        self.fc1 = None
        self.fc2 = None

        self.act = nn.ReLU(inplace=True)

        self.conv_list1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_features, self.in_features, groups=self.in_features, **self.conv_args),
                nn.BatchNorm2d(self.in_features),
            ),
            nn.Sequential(
                nn.Conv2d(self.in_features, self.in_features, 1, 1, 0, groups=self.in_features, bias=False),
                nn.BatchNorm2d(self.in_features),
            ),
            nn.Identity(),
        ])
        self.fc_list1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_features, self.hidden_features, 1, 1, 0, bias=False),
                nn.BatchNorm2d(self.hidden_features),
            ) for _ in range(n_blocks)
        ])
        self.fc_list2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.hidden_features, self.in_features, 1, 1, 0, bias=False),
                nn.BatchNorm2d(self.in_features),
            ) for _ in range(n_blocks)
        ])

    def forward(self, x):
        x = rearrange(x, 'b (n1 n2) c -> b c n1 n2', n1=self.N, n2=self.N)
        if self.conv1:
            x = self.act(self.conv1(x))
            x = self.act(self.fc1(x))
            out = self.fc2(x)
        else:
            xs = self.substitute(x.unsqueeze(-1), self.conv_list1, 0, 0, self.training)
            xs = self.guided_activation(xs)
            xs = self.substitute(xs, self.fc_list1, 0, 0, self.training)
            xs = self.guided_activation(xs)
            xs = self.substitute(xs, self.fc_list2, 0, 0, self.training)
            out = xs.sum(-1)
        return rearrange(out, 'b c n1 n2 -> b (n1 n2) c')

    def substitute(self, x, fc_layer, shuffle, neural_drop_rate, training):
        n_x = x.size(-1)
        n_conv = len(fc_layer)
        feature_shape = list(x.size()[1:-1])
        x_out = list()

        x = x.permute(4, 0, 1, 2, 3).reshape(-1, *feature_shape)

        for fc in fc_layer:
            x_out.append(fc(x))

        x_out = torch.cat(x_out, dim=0)
        x_out = x_out.reshape(n_x * n_conv, -1, *list(x_out.size()[1:]))

        if training:
            x_out = x_out[torch.randperm(x_out.size(0))]
        x_out = x_out.reshape(n_conv, n_x, *list(x_out.size()[1:]))

        return x_out.sum(1).permute(1, 2, 3, 4, 0)

    def guided_activation(self, xs):
        x = torch.sum(xs, dim=4).squeeze(-1)
        x = self.act(x)

        dead_idx = x == 0
        xs[dead_idx] = 0
        return xs

    def re_parameterization_list(self, blocks, in_features, out_features, scale, **conv_args):
        conv_args['bias'] = True
        groups = in_features if in_features == out_features else 1
        conv = nn.Conv2d(in_features, out_features, groups=groups, **conv_args)
        if conv.kernel_size == (3, 3):
            eq_k, eq_b = get_equivalent_kernel_bias(blocks, scale)
        else:
            eq_k, eq_b = fuse_fc(blocks, scale)

        conv.weight.data = eq_k
        conv.bias.data = eq_b

        return conv

    def re_parameterization(self):
        self.conv1 = self.re_parameterization_list(self.conv_list1, self.in_features, self.in_features, 1,
                                                   **self.conv_args)
        self.fc1 = self.re_parameterization_list(self.fc_list1, self.in_features, self.hidden_features, self.n_blocks,
                                                 kernel_size=(1, 1), bias=True)

        self.conv2 = self.re_parameterization_list(self.conv_list2, self.hidden_features, self.hidden_features,
                                                   self.n_blocks, **self.conv_args)
        self.fc2 = self.re_parameterization_list(self.fc_list2, self.in_features, self.hidden_features, self.n_blocks,
                                                 kernel_size=(1, 1), bias=True)

        self.__delattr__('conv_list1')
        self.__delattr__('fc_list1')
        self.__delattr__('conv_list2')
        self.__delattr__('fc_list2')


class AddMlpV5(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            n_blocks=3,
            N=14,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features or in_features
        self.hidden_features = int(in_features * 4)
        self.n_blocks = n_blocks
        self.N = N
        self.conv_args = {
            'kernel_size': (3, 3),
            'stride': 1,
            'padding': 1,
            'bias': False,
        }

        self.conv1 = None
        self.fc1 = None
        self.fc2 = None

        self.act = nn.ReLU(inplace=True)

        self.conv_list1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_features, self.in_features, groups=self.in_features, **self.conv_args),
                nn.BatchNorm2d(self.in_features),
            ),
            nn.Sequential(
                nn.Conv2d(self.in_features, self.in_features, 1, 1, 0, groups=self.in_features, bias=False),
                nn.BatchNorm2d(self.in_features),
            ),
            nn.Identity(),
        ])
        self.fc_list1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_features, self.hidden_features, 1, 1, 0, bias=False),
                nn.BatchNorm2d(self.hidden_features),
            ) for _ in range(n_blocks)
        ])
        self.fc_list2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.hidden_features, self.in_features, 1, 1, 0, bias=False),
                nn.BatchNorm2d(self.in_features),
            ) for _ in range(n_blocks)
        ])

    def forward(self, x):
        x = rearrange(x, 'b (n1 n2) c -> b c n1 n2', n1=self.N, n2=self.N)
        if self.conv1:
            x = self.act(self.conv1(x))
            x = self.act(self.fc1(x))
            out = self.fc2(x)
        else:
            x = self.adding(x, self.conv_list1)
            x = self.act(x)
            x = self.adding(x, self.fc_list1)
            x = self.act(x)
            out = self.adding(x, self.fc_list2)

        return rearrange(out, 'b c n1 n2 -> b (n1 n2) c')

    def adding(self, x, fc_layer):
        x_out = 0

        for fc in fc_layer:
            x_out = x_out + fc(x)

        return x_out


class ConvMlpV5(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            n_blocks=3,
            N=14,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features or in_features
        self.hidden_features = int(in_features * 4)
        self.n_blocks = n_blocks
        self.N = N
        self.conv_args = {
            'kernel_size': (3, 3),
            'stride': 1,
            'padding': 1,
            'bias': False,
        }

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_features, self.in_features, groups=self.in_features, **self.conv_args),
            nn.BatchNorm2d(self.in_features),
        )
        self.fc1 = nn.Sequential(
            nn.Conv2d(self.in_features, self.hidden_features, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.hidden_features),
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(self.hidden_features, self.in_features, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.in_features),
        )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = rearrange(x, 'b (n1 n2) c -> b c n1 n2', n1=self.N, n2=self.N)

        x = self.act(self.conv1(x))
        x = self.act(self.fc1(x))
        out = self.fc2(x)

        return rearrange(out, 'b c n1 n2 -> b (n1 n2) c')


# For FC
# if __name__ == '__main__':
#     block = SubLinear(5, 10, FA=3)
#     n_param = sum(p.numel() for p in block.parameters() if p.requires_grad)
#     input = torch.rand(2, 5, 3)
#
#     block.eval()
#
#     out = block(input.permute(2, 0, 1).reshape(-1, 5), input.size(-1))
#     block.re_parameterization()
#     reparm_out = block(input.sum(-1))
#     print(out.shape)
#     print(reparm_out.shape)
#     print(((out - reparm_out) ** 2).sum())
#
#     n_reparam = sum(p.numel() for p in block.parameters() if p.requires_grad)
#
#     print(n_param, n_reparam)

# For Conv
if __name__ == '__main__':
    # block = SubInceptionV4Block(5, 5, (3, 3), 2, padding=1, stochastic=1.0)
    # n_param = sum(p.numel() for p in block.parameters() if p.requires_grad)
    # input = torch.rand(2, 5, 32, 32, 3)
    #
    # block.eval()
    #
    # out = block(input).sum(-1)
    # block.re_parameterization()
    # reparm_out = block(input.sum(-1))
    # print(out.shape, reparm_out.shape)
    # print(((out - reparm_out) ** 2).sum())
    #
    # n_reparam = sum(p.numel() for p in block.parameters() if p.requires_grad)
    #
    # print(n_param, n_reparam, sum(p.numel() for p in nn.Conv2d(5, 5, (3, 3), 2).parameters() if p.requires_grad))
    mlp = SubMlpV5(10)
    n_param = sum(p.numel() for p in mlp.parameters() if p.requires_grad)
    mlp.eval()
    input = torch.rand(2, 196, 10)

    out = mlp(input)
    mlp.re_parameterization()
    re_out = mlp(input)

    n_reparam = sum(p.numel() for p in mlp.parameters() if p.requires_grad)
    print(((out - re_out) ** 2).sum())
    print(n_param, n_reparam)
