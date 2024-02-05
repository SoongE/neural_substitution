""" MaxVit and CoAtNet Vision Transformer - CNN Hybrids in PyTorch

This is a from-scratch implementation of both CoAtNet and MaxVit in PyTorch.

99% of the implementation was done from papers, however last minute some adjustments were made
based on the (as yet unfinished?) public code release https://github.com/google-research/maxvit

There are multiple sets of models defined for both architectures. Typically, names with a
 `_rw` suffix are my own original configs prior to referencing https://github.com/google-research/maxvit.
These configs work well and appear to be a bit faster / lower resource than the paper.

The models without extra prefix / suffix' (coatnet_0_224, maxvit_tiny_224, etc), are intended to
match paper, BUT, without any official pretrained weights it's difficult to confirm a 100% match.

Papers:

MaxViT: Multi-Axis Vision Transformer - https://arxiv.org/abs/2204.01697
@article{tu2022maxvit,
  title={MaxViT: Multi-Axis Vision Transformer},
  author={Tu, Zhengzhong and Talebi, Hossein and Zhang, Han and Yang, Feng and Milanfar, Peyman and Bovik, Alan and Li, Yinxiao},
  journal={ECCV},
  year={2022},
}

CoAtNet: Marrying Convolution and Attention for All Data Sizes - https://arxiv.org/abs/2106.04803
@article{DBLP:journals/corr/abs-2106-04803,
  author    = {Zihang Dai and Hanxiao Liu and Quoc V. Le and Mingxing Tan},
  title     = {CoAtNet: Marrying Convolution and Attention for All Data Sizes},
  journal   = {CoRR},
  volume    = {abs/2106.04803},
  year      = {2021}
}

Hacked together by / Copyright 2022, Ross Wightman
"""

import math
from abc import abstractmethod
from functools import partial
from typing import Optional, Tuple
import torch

from dataclasses import dataclass
from timm.layers import DropPath, get_norm_act_layer
from timm.layers import create_attn
from timm.layers import drop_path
from timm.layers import trunc_normal_tf_, make_divisible
from timm.layers.padding import get_padding_value
from timm.models import named_apply
from torch import nn

from src.models.blocks import fuse_bn, get_equivalent_kernel_bias, expend_kernel, merge_1x1_kxk, BNAndPadLayer


@dataclass
class MaxxVitConvCfg:
    block_type: str = 'mbconv'
    expand_ratio: float = 4.0
    expand_output: bool = True  # calculate expansion channels from output (vs input chs)
    kernel_size: int = 3
    group_size: int = 1  # 1 == depthwise
    pre_norm_act: bool = False  # activation after pre-norm
    output_bias: bool = True  # bias for shortcut + final 1x1 projection conv
    stride_mode: str = 'dw'  # stride done via one of 'pool', '1x1', 'dw'
    pool_type: str = 'avg2'
    downsample_pool_type: str = 'avg2'
    padding: str = 'same'
    attn_early: bool = False  # apply attn between conv2 and norm2, instead of after norm2
    attn_layer: str = 'se'
    attn_act_layer: str = 'silu'
    attn_ratio: float = 0.25
    init_values: Optional[float] = 1e-6  # for ConvNeXt block, ignored by MBConv
    act_layer: str = 'gelu_tanh'
    norm_layer: str = ''
    norm_layer_cl: str = ''
    norm_eps: Optional[float] = None

    def __post_init__(self):
        # mbconv vs convnext blocks have different defaults, set in post_init to avoid explicit config args
        assert self.block_type in ('mbconv', 'convnext')
        use_mbconv = self.block_type == 'mbconv'
        if not self.norm_layer:
            self.norm_layer = 'batchnorm2d' if use_mbconv else 'layernorm2d'
        if not self.norm_layer_cl and not use_mbconv:
            self.norm_layer_cl = 'layernorm'
        if self.norm_eps is None:
            self.norm_eps = 1e-5 if use_mbconv else 1e-6
        self.downsample_pool_type = self.downsample_pool_type or self.pool_type


def _init_conv(module, name, scheme=''):
    if isinstance(module, nn.Conv2d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def num_groups(group_size, channels):
    if not group_size:  # 0 or None
        return 1  # normal conv with 1 group
    else:
        # NOTE group_size == 1 -> depthwise conv
        assert channels % group_size == 0
        return channels // group_size


class _SubstitutionABC(nn.Module):
    def __init__(self):
        super().__init__()
        self._is_deploy = False
        self.neural_drop_rate = 0.0
        self.deploy_blocks = None
        self.stochastic = True
        self.n_flow = 1

    @abstractmethod
    def re_parameterization(self):
        """
        Re-parameterize all blocks to single block for deploy.
        """
        # Do re-parameterize
        self.deploy_blocks = nn.Identity()
        self._is_deploy = True

    def forward(self, x):
        if self._is_deploy:
            return self.deploy_blocks(x)
        self.n_flow = x.size(-1)
        return self.substitute(x, self.blocks.values())

    def substitute(self, x, conv_layer):
        """
        x: torch.Tensor of shape [B, C, H, W, N] / batch_size, channel, height, width, num_blocks
        """
        n_batch = x.size(0)
        n_x = x.size(-1)
        n_conv = len(conv_layer)

        x_out = [0] * n_conv
        if self.training and self.stochastic:
            rand_idx = torch.randperm(n_conv * n_x) % n_conv
        else:
            rand_idx = torch.arange(n_conv * n_x) % n_conv
        p_idx = 0

        x = x.permute(4, 0, 1, 2, 3).flatten(0, 1)
        for conv in conv_layer:
            _out = conv(x).unflatten(0, (n_x, n_batch))
            _out = drop_path(_out, self.neural_drop_rate, self.training)

            for i, idx in enumerate(rand_idx[p_idx * n_x: (p_idx + 1) * n_x]):
                x_out[idx] = x_out[idx] + _out[i]
            p_idx = p_idx + 1

        x_out = torch.stack(x_out, dim=0)
        return x_out.permute(1, 2, 3, 4, 0)


def activation_for_substitution(xs):
    x = xs.sum(-1)
    x = torch.nn.functional.relu(x)
    dead_idx = (x != 0).float()

    xs = torch.mul(xs, dead_idx.unsqueeze(-1))
    return xs


class SubMbConvBlock(nn.Module):
    """ Pre-Norm Conv Block - 1x1 - kxk - 1x1, w/ inverted bottleneck (expand)
    """

    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            stride: int = 1,
            dilation: Tuple[int, int] = (1, 1),
            cfg: MaxxVitConvCfg = MaxxVitConvCfg(),
            drop_path: float = 0.,
            n: int = 3,
    ):
        super(SubMbConvBlock, self).__init__()
        mid_chs = make_divisible((out_chs if cfg.expand_output else in_chs) * cfg.expand_ratio)
        groups = num_groups(cfg.group_size, mid_chs)
        self.n = n

        if stride == 2:
            self.shortcut = SubDownsample2d(
                in_chs, out_chs, pool_type=cfg.pool_type, bias=cfg.output_bias, padding=cfg.padding, n=n)
        else:
            self.shortcut = nn.Identity()

        assert cfg.stride_mode in ('pool', '1x1', 'dw')
        stride_pool, stride_1, stride_2 = 1, 1, 1
        if cfg.stride_mode == 'pool':
            # NOTE this is not described in paper, experiment to find faster option that doesn't stride in 1x1
            stride_pool, dilation_2 = stride, dilation[1]
            # FIXME handle dilation of avg pool
        elif cfg.stride_mode == '1x1':
            # NOTE I don't like this option described in paper, 1x1 w/ stride throws info away
            stride_1, dilation_2 = stride, dilation[1]
        else:
            stride_2, dilation_2 = stride, dilation[0]

        self.pre_norm = SubBatchNorm2d(in_chs, n)
        # norm_act_layer = partial(get_norm_act_layer(cfg.norm_layer, cfg.act_layer), eps=cfg.norm_eps)
        # self.pre_norm = norm_act_layer(in_chs, apply_act=cfg.pre_norm_act)
        if stride_pool > 1:
            self.down = SubDownsample2d(in_chs, in_chs, pool_type=cfg.downsample_pool_type, padding=cfg.padding, n=n)
        else:
            self.down = nn.Identity()
        self.conv1_1x1 = SubConvBNBlock(in_chs, mid_chs, 1, stride=stride_1, n=n)

        padding, _ = get_padding_value(cfg.padding, cfg.kernel_size)
        self.conv2_kxk = SubConvBNBlock(mid_chs, mid_chs, cfg.kernel_size, stride=stride_2, dilation=dilation_2,
                                        groups=groups, padding=padding, n=n)

        attn_kwargs = {}
        if isinstance(cfg.attn_layer, str):
            if cfg.attn_layer == 'se' or cfg.attn_layer == 'eca':
                attn_kwargs['act_layer'] = cfg.attn_act_layer
                attn_kwargs['rd_channels'] = int(cfg.attn_ratio * (out_chs if cfg.expand_output else mid_chs))

        self.se = create_attn(cfg.attn_layer, mid_chs, **attn_kwargs)

        self.conv3_1x1 = SubConvBNBlock(mid_chs, out_chs, 1, bias=False, n=n)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.is_deploy = False

    def init_weights(self, scheme=''):
        named_apply(partial(_init_conv, scheme=scheme), self)

    def forward(self, x):
        if self.is_deploy:
            return self.deploy_forward(x)

        x = x.unsqueeze(-1)
        shortcut = self.shortcut(x)
        x = self.pre_norm(x)
        x = self.down(x)

        # 1x1 expansion conv & norm-act
        x = self.conv1_1x1(x)
        x = activation_for_substitution(x)

        # depthwise / grouped 3x3 conv w/ SE (or other) channel attention & norm-act
        x = self.conv2_kxk(x)
        x = activation_for_substitution(x)

        if self.se is not None:
            x = x.sum(-1)
            x = self.se(x)
            x = x.unsqueeze(-1)

        # 1x1 linear projection to output width
        x = self.conv3_1x1(x)

        if shortcut.size()[-1] == 1:
            shortcut = (shortcut / x.size(-1)).repeat(1, 1, 1, 1, x.size(-1))

        x = self.drop_path(x) + shortcut
        return x.sum(-1)

    def deploy_forward(self, x):
        shortcut = self.shortcut(x)
        x = self.pre_norm(x)
        x = self.down(x)

        # 1x1 expansion conv & norm-act
        x = self.conv1_1x1(x)
        x = torch.nn.functional.relu(x)

        # depthwise / grouped 3x3 conv w/ SE (or other) channel attention & norm-act
        x = self.conv2_kxk(x)
        x = torch.nn.functional.relu(x)

        if self.se is not None:
            x = self.se(x)

        # 1x1 linear projection to output width
        x = self.conv3_1x1(x)
        x = self.drop_path(x) + shortcut
        return x

    def re_parameterization(self):
        self.is_deploy = True


class SubMbConvBlockV7(nn.Module):
    """ Pre-Norm Conv Block - 1x1 - kxk - 1x1, w/ inverted bottleneck (expand)
    """

    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            stride: int = 1,
            dilation: Tuple[int, int] = (1, 1),
            cfg: MaxxVitConvCfg = MaxxVitConvCfg(),
            drop_path: float = 0.,
            n: int = 3,
    ):
        super(SubMbConvBlockV7, self).__init__()
        norm_act_layer = partial(get_norm_act_layer(cfg.norm_layer, cfg.act_layer), eps=cfg.norm_eps)
        mid_chs = make_divisible((out_chs if cfg.expand_output else in_chs) * cfg.expand_ratio)
        groups = num_groups(cfg.group_size, mid_chs)
        self.n = n

        if stride == 2:
            self.shortcut = SubDownsample2d(
                in_chs, out_chs, pool_type=cfg.pool_type, bias=cfg.output_bias, padding=cfg.padding, n=n)
        else:
            self.shortcut = nn.Identity()

        assert cfg.stride_mode in ('pool', '1x1', 'dw')
        stride_pool, stride_1, stride_2 = 1, 1, 1
        if cfg.stride_mode == 'pool':
            # NOTE this is not described in paper, experiment to find faster option that doesn't stride in 1x1
            stride_pool, dilation_2 = stride, dilation[1]
            # FIXME handle dilation of avg pool
        elif cfg.stride_mode == '1x1':
            # NOTE I don't like this option described in paper, 1x1 w/ stride throws info away
            stride_1, dilation_2 = stride, dilation[1]
        else:
            stride_2, dilation_2 = stride, dilation[0]

        # self.pre_norm = SubBatchNorm2d(in_chs, n)
        # norm_act_layer = partial(get_norm_act_layer(cfg.norm_layer, cfg.act_layer), eps=cfg.norm_eps)
        self.pre_norm = norm_act_layer(in_chs, apply_act=cfg.pre_norm_act)
        if stride_pool > 1:
            self.down = SubDownsample2d(in_chs, in_chs, pool_type=cfg.downsample_pool_type, padding=cfg.padding, n=n)
        else:
            self.down = nn.Identity()
        self.conv1_1x1 = SubInceptionV7Block(in_chs, mid_chs, 1, stride=stride_1)

        padding, _ = get_padding_value(cfg.padding, cfg.kernel_size)
        self.conv2_kxk = SubInceptionV7Block(mid_chs, mid_chs, cfg.kernel_size, stride=stride_2, dilation=dilation_2,
                                             groups=groups, padding=padding)

        attn_kwargs = {}
        if isinstance(cfg.attn_layer, str):
            if cfg.attn_layer == 'se' or cfg.attn_layer == 'eca':
                attn_kwargs['act_layer'] = cfg.attn_act_layer
                attn_kwargs['rd_channels'] = int(cfg.attn_ratio * (out_chs if cfg.expand_output else mid_chs))

        self.se = create_attn(cfg.attn_layer, mid_chs, **attn_kwargs)

        self.conv3_1x1 = SubInceptionV7Block(mid_chs, out_chs, 1, bias=False, )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.is_deploy = False

        self.down.stochastic=False
        self.conv1_1x1=False
        self.conv2_kxk=False

    def init_weights(self, scheme=''):
        named_apply(partial(_init_conv, scheme=scheme), self)

    def forward(self, x):
        if self.is_deploy:
            return self.deploy_forward(x)

        x = x.unsqueeze(-1)
        shortcut = self.shortcut(x)
        x = self.pre_norm(x.squeeze(-1))
        x = self.down(x.unsqueeze(-1))

        # 1x1 expansion conv & norm-act
        x = self.conv1_1x1(x)
        x = activation_for_substitution(x)

        # depthwise / grouped 3x3 conv w/ SE (or other) channel attention & norm-act
        x = self.conv2_kxk(x)
        x = activation_for_substitution(x)

        if self.se is not None:
            x = x.sum(-1)
            x = self.se(x)
            x = x.unsqueeze(-1)

        # 1x1 linear projection to output width
        x = self.conv3_1x1(x)

        if shortcut.size()[-1] == 1:
            shortcut = (shortcut / x.size(-1)).repeat(1, 1, 1, 1, x.size(-1))

        x = self.drop_path(x) + shortcut
        return x.sum(-1)

    def deploy_forward(self, x):
        shortcut = self.shortcut(x)
        x = self.pre_norm(x)
        x = self.down(x)

        # 1x1 expansion conv & norm-act
        x = self.conv1_1x1(x)
        x = torch.nn.functional.relu(x)

        # depthwise / grouped 3x3 conv w/ SE (or other) channel attention & norm-act
        x = self.conv2_kxk(x)
        x = torch.nn.functional.relu(x)

        if self.se is not None:
            x = self.se(x)

        # 1x1 linear projection to output width
        x = self.conv3_1x1(x)
        x = self.drop_path(x) + shortcut
        return x

    def re_parameterization(self):
        self.is_deploy = True


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


class SubBatchNorm2d(_SubstitutionABC):
    def __init__(self, in_chs, n=1, **bn_kwargs):
        super().__init__()
        self.blocks = nn.ModuleDict()
        self.n = n
        for i in range(n):
            self.blocks.add_module(f'bn{i}', nn.BatchNorm2d(in_chs, **bn_kwargs))

    def re_parameterization(self):
        _f = list(self.blocks.values())[0]
        self.deploy_blocks = nn.Conv2d(_f.num_features, _f.num_features, (1, 1))
        w, b = 0, 0
        for layer in self.blocks.values():
            _w, _b = fuse_only_bn(layer, (1, 1), self.n_flow)
            w = w + _w
            b = b + _b

        self.deploy_blocks.weight.data = w
        self.deploy_blocks.bias.data = b
        self._is_deploy = True
        self.__delattr__('blocks')


class Divide(nn.Module):
    def __init__(self, flow):
        super().__init__()
        self.flow = flow

    def forward(self, x):
        return x / self.flow


class SubDownsample2d(_SubstitutionABC):
    def __init__(
            self,
            dim: int,
            dim_out: int,
            pool_type: str = 'avg2',
            padding: str = '',
            bias: bool = True,
            n: int = 1,
    ):
        super().__init__()
        assert pool_type in ('avg2')
        self.n = n
        self.blocks = nn.ModuleDict()
        self.down_dim = dim != dim_out
        self.dim = dim

        padding, is_dynamic = get_padding_value(padding, 1, stride=2)
        self.padding = padding

        if not self.down_dim:
            for i in range(n):
                self.blocks.add_module(f'1x1-avg{i}', nn.Sequential(
                    Divide(n),
                    nn.AvgPool2d(kernel_size=2, stride=2),
                ))
        else:
            for i in range(n):
                self.blocks.add_module(f'1x1-avg{i}', nn.Sequential(
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(dim, dim_out, kernel_size=(1, 1), bias=False, padding=self.padding),
                    nn.BatchNorm2d(dim_out),
                ))

    def re_parameterization(self):
        if not self.down_dim:
            self.deploy_blocks = nn.AvgPool2d(2, 2)

        else:
            _f = list(self.blocks.values())[0][1]
            conv = nn.Conv2d(_f.in_channels, _f.out_channels, (1, 1), 1, padding=self.padding)
            w, b = 0, 0
            for layer in self.blocks.values():
                _k3, _b3 = fuse_bn(*layer[1:], self.n_flow)
                w = w + _k3
                b = b + _b3

            conv.weight.data = w
            conv.bias.data = b
            self.deploy_blocks = nn.Sequential(
                nn.AvgPool2d(2, 2),
                conv,
            )
        self._is_deploy = True
        self.__delattr__('blocks')


class SubConvBNBlock(_SubstitutionABC):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False,
                 bn=nn.BatchNorm2d, groups=1, n=1, **kwargs):
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
        self.n = n
        self.blocks = nn.ModuleDict()

        for i in range(n):
            self.blocks.add_module(f'kxk_{i}', nn.Sequential(
                nn.Conv2d(**self.conv_args),
                bn(out_channels),
            ))

    def re_parameterization(self):
        self.conv_args['bias'] = True
        self.deploy_blocks = nn.Conv2d(**self.conv_args)

        eq_k, eq_b = get_equivalent_kernel_bias(list(self.blocks.values()), self.n_flow)

        self.deploy_blocks.weight.data = eq_k
        self.deploy_blocks.bias.data = eq_b
        self._is_deploy = True
        self.__delattr__('blocks')


class SubInceptionV7Block(_SubstitutionABC):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, bn=nn.BatchNorm2d,
                 **kwargs):
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
        hidden_channels = int(in_channels * 2)
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

        self.blocks.update({'dsx2': nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=(1, 1), bias=False, groups=kwargs.get('groups', 1)),
            BNAndPadLayer(padding, hidden_channels),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False,
                      groups=kwargs.get('groups', 1)),
            bn(out_channels),
        )})

    def re_parameterization(self):
        self.conv_args['bias'] = True
        self.deploy_blocks = nn.Conv2d(**self.conv_args)

        _k0, _b0 = fuse_bn(*self.blocks['kxk'], self.n_flow)

        _k1, _b1 = fuse_bn(*self.blocks['1x1'], self.n_flow)
        _k1 = expend_kernel(_k1, self.conv_args['kernel_size'])

        _k3, _b3 = fuse_bn(*self.blocks['dsx2'][:2], self.n_flow)
        _k33, _b33 = fuse_bn(*self.blocks['dsx2'][2:], self.n_flow)
        _k3, _b3 = merge_1x1_kxk(_k3, _b3, _k33, _b33, self.conv_args.get('groups', 1))

        self.deploy_blocks.weight.data = sum([_k0, _k3, _k1])
        self.deploy_blocks.bias.data = sum([_b0, _b3, _b1])
        self._is_deploy = True
        self.__delattr__('blocks')


if __name__ == '__main__':
    import torch

    block = SubMbConvBlockV7(10, 10, stride=1)
    for m in block.modules():
        if isinstance(m, nn.BatchNorm2d):
            nn.init.uniform_(m.running_mean, 0, 0.1)
            nn.init.uniform_(m.running_var, 0, 0.1)
            nn.init.uniform_(m.weight, 0, 0.1)
            nn.init.uniform_(m.bias, 0, 0.1)
    block.eval()
    x = torch.rand(2, 10, 32, 32)
    out1 = block(x)

    for m in block.modules():
        if hasattr(m, 're_parameterization'):
            m.re_parameterization()

    out2 = block(x)
    print(out1.shape, out2.shape)
    print(f"Diff: ", ((out1 - out2) ** 2).sum().item())

    # n_block, in_channel = 3, 10
    # x = torch.rand(2, in_channel, 14, 14, n_block)
    # block = SubDownsample2d(in_channel, 20)
    # block.eval()
    #
    # if x.dim() == 4:
    #     x = x.unsqueeze(-1)
    # with torch.no_grad():
    #     out1 = block(x)
    #     block.re_parameterization()
    #     out2 = block(x.sum(-1))
    #
    # out1 = out1.sum(-1)
    # print(f"Diff: ", ((out1 - out2) ** 2).sum().item())
