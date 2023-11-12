import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    n_x = x.size(-1)
    n_conv = len(conv_layer)
    feature_shape = list(x.size()[1:-1])
    x_out = list()

    x = x.permute(4, 0, 1, 2, 3).reshape(-1, *feature_shape)

    for conv in conv_layer:
        x_out.append(conv(x))

    x_out = torch.cat(x_out, dim=0)
    x_out = x_out.reshape(n_x * n_conv, -1, *list(x_out.size()[1:]))

    if training:
        # if shuffle > random.random():
        x_out = x_out[torch.randperm(x_out.size(0))]
        # x_out = drop_path(x_out, neural_drop_rate, training)
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
    block = SubInceptionV1Block(5, 10, (3, 3), 3, padding=1, stochastic=1.0)
    n_param = sum(p.numel() for p in block.parameters() if p.requires_grad)
    input = torch.rand(2, 5, 32, 32, 3)

    block.eval()

    out = block(input).sum(-1)
    block.re_parameterization()
    reparm_out = block(input.sum(-1))
    print(((out - reparm_out) ** 2).sum())

    n_reparam = sum(p.numel() for p in block.parameters() if p.requires_grad)

    print(n_param, n_reparam)
