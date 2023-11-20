#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
import random
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['MobileOne', 'mobileone', 'reparameterize_model']

from timm.layers import drop_path

from timm.models import register_model

from src.models.blocks import BNAndPadLayer, fuse_bn, merge_1x1_kxk


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

    randidx = torch.randperm(x_out.size(0))
    if training:
        if shuffle > random.random():
            x_out = x_out[randidx]
        x_out = drop_path(x_out, neural_drop_rate, training)
    x_out = x_out.reshape(n_conv, n_x, *list(x_out.size()[1:]))

    return x_out.sum(1).permute(1, 2, 3, 4, 0)


def activation_for_substitute(xs, x):
    dead_idx = (x != 0).float()
    xs = torch.mul(xs, dead_idx.unsqueeze(-1))
    return xs


class SEBlock(nn.Module):
    """ Squeeze and Excite module.

        Pytorch implementation of `Squeeze-and-Excitation Networks` -
        https://arxiv.org/pdf/1709.01507.pdf
    """

    def __init__(self,
                 in_channels: int,
                 rd_ratio: float = 0.0625) -> None:
        """ Construct a Squeeze and Excite Module.

        :param in_channels: Number of input channels.
        :param rd_ratio: Input channel reduction ratio.
        """
        super(SEBlock, self).__init__()
        self.reduce = nn.Conv2d(in_channels=in_channels,
                                out_channels=int(in_channels * rd_ratio),
                                kernel_size=1,
                                stride=1,
                                bias=True)
        self.expand = nn.Conv2d(in_channels=int(in_channels * rd_ratio),
                                out_channels=in_channels,
                                kernel_size=1,
                                stride=1,
                                bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """ Apply forward pass. """
        b, c, h, w = inputs.size()
        x = F.avg_pool2d(inputs, kernel_size=[h, w])
        x = self.reduce(x)
        x = F.relu(x)
        x = self.expand(x)
        x = torch.sigmoid(x)
        x = x.view(-1, c, 1, 1)
        return inputs * x


class MobileOneBlock(nn.Module):
    """ MobileOne building block.

        This block has a multi-branched architecture at train-time
        and plain-CNN style architecture at inference time
        For more details, please refer to our paper:
        `An Improved One millisecond Mobile Backbone` -
        https://arxiv.org/pdf/2206.04040.pdf
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 inference_mode: bool = False,
                 use_se: bool = False,
                 num_conv_branches: int = 1,
                 substitution: bool = False,
                 stochastic=1.0,
                 neural_drop_rate=0.0,
                 ) -> None:
        """ Construct a MobileOneBlock module.

        :param in_channels: Number of channels in the input.
        :param out_channels: Number of channels produced by the block.
        :param kernel_size: Size of the convolution kernel.
        :param stride: Stride size.
        :param padding: Zero-padding size.
        :param dilation: Kernel dilation factor.
        :param groups: Group number.
        :param inference_mode: If True, instantiates model in inference mode.
        :param use_se: Whether to use SE-ReLU activations.
        :param num_conv_branches: Number of linear conv branches.
        """
        super(MobileOneBlock, self).__init__()
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches
        self.padding = padding
        self.dilation = dilation
        self.substitution = substitution
        self.stochastic = stochastic
        self.neural_drop_rate = neural_drop_rate
        self.n_flow = 1

        # Check if SE-ReLU is requested
        if use_se:
            self.se = SEBlock(out_channels)
        else:
            self.se = nn.Identity()
        self.activation = nn.ReLU()

        if inference_mode:
            self.reparam_conv = nn.Conv2d(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          groups=groups,
                                          bias=True)
        else:
            # Re-parameterizable skip connection
            self.rbr_skip = nn.BatchNorm2d(num_features=in_channels) \
                if out_channels == in_channels and stride == 1 else None

            # Re-parameterizable conv branches
            rbr_conv = list()
            if self.num_conv_branches == 1:
                rbr_conv.append(self._conv_bn(kernel_size=kernel_size, padding=padding))
            else:
                hidden_channels1 = int(in_channels * 2)
                hidden_channels2 = int(in_channels * 4)

                rbr_conv.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_channels1, kernel_size=(1, 1), bias=False, groups=groups),
                    BNAndPadLayer(padding, hidden_channels1),
                    nn.Conv2d(hidden_channels1, out_channels, kernel_size=kernel_size, stride=stride, bias=False,
                              groups=groups),
                    nn.BatchNorm2d(out_channels),
                ))
                rbr_conv.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_channels2, kernel_size=(1, 1), bias=False, groups=groups),
                    BNAndPadLayer(padding, hidden_channels2),
                    nn.Conv2d(hidden_channels2, out_channels, kernel_size=kernel_size, stride=stride, bias=False,
                              groups=groups),
                    nn.BatchNorm2d(out_channels),
                ))

            self.rbr_conv = nn.ModuleList(rbr_conv)

            # Re-parameterizable scale branch
            self.rbr_scale = None
            if kernel_size > 1:
                self.rbr_scale = self._conv_bn(kernel_size=1,
                                               padding=0)

            if self.substitution:
                self.sub_act = nn.ReLU()
                self.layers = nn.ModuleList()
                self.layers.extend(self.rbr_conv)
                if self.rbr_scale:
                    self.layers.append(self.rbr_scale)
                if self.rbr_skip:
                    self.layers.append(self.rbr_skip)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Apply forward pass. """
        # Inference mode forward pass.
        if self.inference_mode:
            return self.activation(self.se(self.reparam_conv(x)))

        # Multi-branched train-time forward pass.
        # Skip branch output
        if self.substitution:
            if x.dim() == 4:
                x = x.unsqueeze(-1)
            self.n_flow = x.size(-1)
            xs = substitute(x, self.layers, self.stochastic, self.neural_drop_rate, self.training)
            _x = torch.mean(xs, dim=4).squeeze(-1)
            _x = self.sub_act(_x)
            xs = activation_for_substitute(xs, _x)
            return xs
        else:
            identity_out = 0
            if self.rbr_skip is not None:
                identity_out = self.rbr_skip(x)

            # Scale branch output
            scale_out = 0
            if self.rbr_scale is not None:
                scale_out = self.rbr_scale(x)

            # Other branches
            out = scale_out + identity_out
            for ix in range(self.num_conv_branches):
                out += self.rbr_conv[ix](x)

            return self.activation(self.se(out))

    def re_parameterization(self):
        """ Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        if self.inference_mode:
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(in_channels=self.in_channels,
                                      out_channels=self.out_channels,
                                      kernel_size=self.kernel_size,
                                      stride=self.stride,
                                      padding=self.padding,
                                      dilation=self.dilation,
                                      groups=self.groups,
                                      bias=True)
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        # Delete un-used branches
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_conv')
        self.__delattr__('rbr_scale')
        if hasattr(self, 'rbr_skip'):
            self.__delattr__('rbr_skip')

        self.inference_mode = True

    def _get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83

        :return: Tuple of (kernel, bias) after fusing branches.
        """
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale, self.n_flow)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale,
                                                   [pad, pad, pad, pad])

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip, self.n_flow)

        # get weights and bias of conv branches
        if self.num_conv_branches == 1:
            kernel_conv, bias_conv = self._fuse_bn_tensor(self.rbr_conv[0], self.n_flow)
        else:
            kernel_conv = 0
            bias_conv = 0
            for ix in range(2):
                _k1, _b1 = fuse_bn(*self.rbr_conv[ix][:2], self.n_flow)
                _k2, _b2 = fuse_bn(*self.rbr_conv[ix][2:], self.n_flow)
                _kernel, _bias = merge_1x1_kxk(_k1, _b1, _k2, _b2, self.groups)

                kernel_conv += _kernel
                bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(self, branch, scale=1) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to fuse batchnorm layer with preceeding conv layer.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95

        :param branch:
        :return: Tuple of (kernel, bias) after fusing batchnorm.
        """
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean * scale
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias * scale
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros((self.in_channels,
                                            input_dim,
                                            self.kernel_size,
                                            self.kernel_size),
                                           dtype=branch.weight.dtype,
                                           device=branch.weight.device)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim,
                                    self.kernel_size // 2,
                                    self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean * scale
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias * scale
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(self,
                 kernel_size: int,
                 padding: int) -> nn.Sequential:
        """ Helper method to construct conv-batchnorm layers.

        :param kernel_size: Size of the convolution kernel.
        :param padding: Zero-padding size.
        :return: Conv-BN module.
        """
        mod_list = nn.Sequential()
        mod_list.add_module('conv', nn.Conv2d(in_channels=self.in_channels,
                                              out_channels=self.out_channels,
                                              kernel_size=kernel_size,
                                              stride=self.stride,
                                              padding=padding,
                                              groups=self.groups,
                                              bias=False))
        mod_list.add_module('bn', nn.BatchNorm2d(num_features=self.out_channels))
        return mod_list


class MobileOne(nn.Module):
    """ MobileOne Model

        Pytorch implementation of `An Improved One millisecond Mobile Backbone` -
        https://arxiv.org/pdf/2206.04040.pdf
    """

    def __init__(self,
                 num_blocks_per_stage: List[int] = [2, 8, 10, 1],
                 num_classes: int = 1000,
                 width_multipliers: Optional[List[float]] = None,
                 inference_mode: bool = False,
                 use_se: bool = False,
                 num_conv_branches: int = 1,
                 substitution: bool = True,
                 **kwargs) -> None:
        """ Construct MobileOne model.

        :param num_blocks_per_stage: List of number of blocks per stage.
        :param num_classes: Number of classes in the dataset.
        :param width_multipliers: List of width multiplier for blocks in a stage.
        :param inference_mode: If True, instantiates model in inference mode.
        :param use_se: Whether to use SE-ReLU activations.
        :param num_conv_branches: Number of linear conv branches.
        """
        super().__init__()

        assert substitution
        assert len(width_multipliers) == 4
        self.inference_mode = inference_mode
        self.in_planes = min(64, int(64 * width_multipliers[0]))
        self.use_se = use_se
        self.num_conv_branches = num_conv_branches
        self.substitution = substitution
        self.stochastic = kwargs.get('stochastic', 1.0)
        self.neural_drop_rate = kwargs.get('neural_drop_rate', 0.0)
        self.block_idx = 0
        self.n_total_blocks = sum(num_blocks_per_stage)

        # Build stages
        if 'cifar' in kwargs.get('stem_type', ''):
            self.stage0 = MobileOneBlock(in_channels=3, out_channels=self.in_planes,
                                         kernel_size=3, stride=1, padding=1,
                                         inference_mode=self.inference_mode, substitution=self.substitution)
        else:
            self.stage0 = MobileOneBlock(in_channels=3, out_channels=self.in_planes,
                                         kernel_size=3, stride=2, padding=1,
                                         inference_mode=self.inference_mode, substitution=self.substitution)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multipliers[0]), num_blocks_per_stage[0],
                                       num_se_blocks=0)
        self.stage2 = self._make_stage(int(128 * width_multipliers[1]), num_blocks_per_stage[1],
                                       num_se_blocks=0)
        self.stage3 = self._make_stage(int(256 * width_multipliers[2]), num_blocks_per_stage[2],
                                       num_se_blocks=int(num_blocks_per_stage[2] // 2) if use_se else 0)
        self.stage4 = self._make_stage(int(512 * width_multipliers[3]), num_blocks_per_stage[3],
                                       num_se_blocks=num_blocks_per_stage[3] if use_se else 0)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(512 * width_multipliers[3]), num_classes)

    def init_weights(self, zero_init_last=True, bn_init=False):
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if bn_init and isinstance(m, nn.BatchNorm2d):
                nn.init.uniform_(m.running_mean, 0, 0.1)
                nn.init.uniform_(m.running_var, 0, 0.1)
                nn.init.uniform_(m.weight, 0, 0.1)
                nn.init.uniform_(m.bias, 0, 0.1)
        if zero_init_last:
            for m in self.modules():
                if hasattr(m, 'zero_init_last'):
                    m.zero_init_last()

    def _make_stage(self,
                    planes: int,
                    num_blocks: int,
                    num_se_blocks: int) -> nn.Sequential:
        """ Build a stage of MobileOne model.

        :param planes: Number of output channels.
        :param num_blocks: Number of blocks in this stage.
        :param num_se_blocks: Number of SE blocks in this stage.
        :return: A stage of MobileOne model.
        """
        # Get strides for all layers
        strides = [2] + [1] * (num_blocks - 1)
        blocks = []

        for ix, stride in enumerate(strides):
            use_se = False
            if num_se_blocks > num_blocks:
                raise ValueError("Number of SE blocks cannot "
                                 "exceed number of layers.")
            if ix >= (num_blocks - num_se_blocks):
                use_se = True
            ndr = self.neural_drop_rate * self.block_idx / (self.n_total_blocks - 1)
            # Depthwise conv
            blocks.append(MobileOneBlock(in_channels=self.in_planes,
                                         out_channels=self.in_planes,
                                         kernel_size=3,
                                         stride=stride,
                                         padding=1,
                                         groups=self.in_planes,
                                         inference_mode=self.inference_mode,
                                         use_se=use_se,
                                         num_conv_branches=self.num_conv_branches,
                                         substitution=self.substitution,
                                         stochastic=self.stochastic,
                                         neural_drop_rate=ndr,
                                         ))
            # Pointwise conv
            blocks.append(MobileOneBlock(in_channels=self.in_planes,
                                         out_channels=planes,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         groups=1,
                                         inference_mode=self.inference_mode,
                                         use_se=use_se,
                                         num_conv_branches=self.num_conv_branches,
                                         substitution=self.substitution,
                                         stochastic=self.stochastic,
                                         neural_drop_rate=ndr,
                                         ))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Apply forward pass. """
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        if x.dim() == 5:
            x = x.sum(-1)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


PARAMS = {
    "s0": {"width_multipliers": (0.75, 1.0, 1.0, 2.0), "num_conv_branches": 2},
    "s1": {"width_multipliers": (1.5, 1.5, 2.0, 2.5)},
    "s2": {"width_multipliers": (1.5, 2.0, 2.5, 4.0)},
    "s3": {"width_multipliers": (2.0, 2.5, 3.0, 4.0)},
    "s4": {"width_multipliers": (3.0, 3.5, 3.5, 4.0), "use_se": True},

    "s0+": {"width_multipliers": (0.75, 1.0, 1.0, 2.0), "num_conv_branches": 2},
    "s1+": {"width_multipliers": (1.5, 1.5, 2.0, 2.5), "num_conv_branches": 2},
}


@register_model
def mobileoneSub(name, num_classes: int = 1000, inference_mode: bool = False,
                 variant: str = "s1+", **kwargs) -> nn.Module:
    """Get MobileOne model.

    :param num_classes: Number of classes in the dataset.
    :param inference_mode: If True, instantiates model in inference mode.
    :param variant: Which type of model to generate.
    :return: MobileOne model. """
    variant_params = PARAMS[variant]
    variant_params.update(kwargs)
    return MobileOne(num_classes=num_classes, inference_mode=inference_mode, **variant_params)


if __name__ == '__main__':
    x = torch.rand(2, 3, 32, 32)
    model = mobileoneSub('', num_classes=100, variant="s0+", substitution=True, stem_type='cifar')
    print(model(x).shape)
