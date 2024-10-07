import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.helpers import build_model_with_cfg, checkpoint_seq
from timm.models.layers import DropBlock2d, DropPath, create_attn, get_act_layer, get_norm_layer, \
    create_classifier
from timm.models.registry import register_model

from src.models.utils import activation_for_substitute
from src.models.blocks import SubConvBNBlock, NS


def get_padding(kernel_size, stride, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def create_aa(aa_layer, channels, stride=2, enable=True):
    if not aa_layer or not enable:
        return nn.Identity()
    return aa_layer(stride) if issubclass(aa_layer, nn.AvgPool2d) else aa_layer(channels=channels, stride=stride)


def downsample_conv(
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        first_dilation=None,
        norm_layer=None,
        sub_block=None,
        n_block=1,
):
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)

    if kernel_size == 3:
        block_fn = sub_block
    else:
        block_fn = SubConvBNBlock
    return block_fn(in_channels, out_channels, kernel_size, stride=stride, n_block=n_block, padding=p,
                    dilation=first_dilation, bn=norm_layer, neural_drop_rate=0.0)


def drop_blocks(drop_prob=0.):
    return [
        None, None,
        partial(DropBlock2d, drop_prob=drop_prob, block_size=5, gamma_scale=0.25) if drop_prob else None,
        partial(DropBlock2d, drop_prob=drop_prob, block_size=3, gamma_scale=1.00) if drop_prob else None]


def make_blocks(
        block_fn,
        channels,
        block_repeats,
        inplanes,
        reduce_first=1,
        output_stride=32,
        down_kernel_size=1,
        avg_down=False,
        drop_block_rate=0.,
        drop_path_rate=0.,
        sub_block=None,
        n_block=1,
        **kwargs,
):
    stages = []
    feature_info = []
    net_num_blocks = sum(block_repeats)
    net_block_idx = 0
    net_stride = 4
    dilation = prev_dilation = 1
    for stage_idx, (planes, num_blocks, db) in enumerate(zip(channels, block_repeats, drop_blocks(drop_block_rate))):
        stage_name = f'layer{stage_idx + 1}'  # never liked this name, but weight compat requires it
        stride = 1 if stage_idx == 0 else 2
        if net_stride >= output_stride:
            dilation *= stride
            stride = 1
        else:
            net_stride *= stride
        downsample = None
        if stride != 1 or inplanes != planes * block_fn.expansion:
            downsample = downsample_conv(
                in_channels=inplanes,
                out_channels=planes * block_fn.expansion,
                kernel_size=down_kernel_size,
                stride=stride,
                dilation=dilation,
                first_dilation=prev_dilation,
                norm_layer=kwargs.get('norm_layer'),
                sub_block=sub_block,
                n_block=n_block,
            )

        block_kwargs = dict(reduce_first=reduce_first, dilation=dilation, drop_block=db, **kwargs)
        blocks = []
        for block_idx in range(num_blocks):
            downsample = downsample if block_idx == 0 else None
            stride = stride if block_idx == 0 else 1
            block_dpr = drop_path_rate * net_block_idx / (net_num_blocks - 1)
            block_kwargs['neural_drop_rate'] = block_kwargs.get('neural_drop_rate', 0.0) * net_block_idx / (
                    net_num_blocks - 1)
            blocks.append(block_fn(
                inplanes, planes, stride, downsample, first_dilation=prev_dilation, sub_block=sub_block,
                n_block=n_block, drop_path=DropPath(block_dpr) if block_dpr > 0. else None,
                **block_kwargs))
            prev_dilation = dilation
            inplanes = planes * block_fn.expansion
            net_block_idx += 1

        stages.append((stage_name, nn.Sequential(*blocks)))
        feature_info.append(dict(num_chs=inplanes, reduction=net_stride, module=stage_name))

    return stages, feature_info


class BottleneckSub(nn.Module):
    expansion = 4

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            downsample=None,
            cardinality=1,
            base_width=64,
            reduce_first=1,
            dilation=1,
            first_dilation=None,
            act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d,
            attn_layer=None,
            aa_layer=None,
            drop_block=None,
            drop_path=None,
            **kwargs,
    ):
        super(BottleneckSub, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        block_fn = kwargs['sub_block']
        n_block = kwargs.get('n_block', None)
        neural_drop_rate = kwargs.get('neural_drop_rate', 0.0)

        self.conv1 = SubConvBNBlock(inplanes, first_planes, kernel_size=1, n_block=n_block,
                                    neural_drop_rate=neural_drop_rate)
        self.act1 = act_layer(inplace=True)

        self.conv2 = block_fn(
            first_planes, width, kernel_size=3, stride=1 if use_aa else stride, n_block=n_block, padding=first_dilation,
            dilation=first_dilation, groups=cardinality, neural_drop_rate=neural_drop_rate)
        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act2 = act_layer(inplace=True)
        self.aa = create_aa(aa_layer, channels=width, stride=stride, enable=use_aa)

        self.conv3 = SubConvBNBlock(width, outplanes, kernel_size=1, n_block=n_block, neural_drop_rate=neural_drop_rate)

        self.se = create_attn(attn_layer, outplanes)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path

        self.re_parameterized = False

    def zero_init_last(self):
        pass
        # if getattr(self.bn3, 'weight', None) is not None:
        #     nn.init.zeros_(self.bn3.weight)

    def train_forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(-1)
        shortcut = x

        xs1 = self.conv1(x)
        x = torch.mean(xs1, dim=4).squeeze(-1)
        x = self.act1(x)
        x = self.aa(x)
        xs1 = activation_for_substitute(xs1, x)

        xs2 = self.conv2(xs1)
        x = torch.mean(xs2, dim=4).squeeze(-1)
        x = self.drop_block(x)
        x = self.act2(x)
        x = self.aa(x)
        xs2 = activation_for_substitute(xs2, x)

        xs3 = self.conv3(xs2)
        x = torch.mean(xs3, dim=4).squeeze(-1)

        if self.se is not None:
            x = self.se(x)
        if self.drop_path is not None:
            x = self.drop_path(x)

        if shortcut.size()[-1] == 1:
            shortcut = (shortcut / xs3.size(-1)).repeat(1, 1, 1, 1, xs3.size(-1))

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)

        x += torch.mean(shortcut, dim=4)
        x = self.act3(x)
        xs3 += shortcut
        xs3 = activation_for_substitute(xs3, x)
        return xs3

    def re_parameterized_forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.drop_block(x)
        x = self.act2(x)
        x = self.aa(x)

        x = self.conv3(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act3(x)
        return x

    def forward(self, x):
        if self.re_parameterized:
            return self.re_parameterized_forward(x)
        return self.train_forward(x)


class BasicBlockSub(nn.Module):
    expansion = 1

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            downsample=None,
            cardinality=1,
            base_width=64,
            reduce_first=1,
            dilation=1,
            first_dilation=None,
            act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d,
            attn_layer=None,
            aa_layer=None,
            drop_block=None,
            drop_path=None,
            **kwargs,
    ):
        super(BasicBlockSub, self).__init__()

        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock does not support changing base width'
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        block_fn = kwargs['sub_block']
        n_block = kwargs.get('n_block', None)
        neural_drop_rate = kwargs.get('neural_drop_rate', 0.0)

        self.conv1 = block_fn(inplanes, first_planes, (3, 3), stride=stride, n_block=n_block, padding=first_dilation,
                              dilation=first_dilation, neural_drop_rate=neural_drop_rate)
        self.conv2 = block_fn(first_planes, outplanes, (3, 3), stride=1, n_block=n_block, padding=dilation,
                              dilation=dilation, neural_drop_rate=neural_drop_rate)

        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act1 = act_layer(inplace=True)
        self.aa = create_aa(aa_layer, channels=first_planes, stride=stride, enable=use_aa)
        self.se = create_attn(attn_layer, outplanes)

        self.act2 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path

        self.re_parameterized = False

    def zero_init_last(self):
        pass

    def re_parameterize(self):
        assert self.re_parameterized is False, f'Re-parameterization already done'
        self.conv1.re_parameterization()
        self.conv2.re_parameterization()
        if self.downsample:
            self.downsample.re_parameterization()
        self.re_parameterized = True

    def train_forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(-1)
        shortcut = x
        xs1 = self.conv1(x)
        x = torch.mean(xs1, dim=4).squeeze(-1)
        x = self.drop_block(x)
        x = self.act2(x)
        x = self.aa(x)
        xs1 = activation_for_substitute(xs1, x)

        xs2 = self.conv2(xs1)
        x = torch.mean(xs2, dim=4).squeeze(-1)
        if self.se is not None:
            x = self.se(x)
        if self.drop_path is not None:
            x = self.drop_path(x)

        if shortcut.size()[-1] == 1:
            shortcut = (shortcut / xs2.size(-1)).repeat(1, 1, 1, 1, xs2.size(-1))

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)

        x += torch.mean(shortcut, dim=4)
        x = self.act2(x)
        xs2 += shortcut
        xs2 = activation_for_substitute(xs2, x)
        return xs2

    def re_parameterized_forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.drop_block(x)
        x = self.act1(x)
        x = self.aa(x)

        x = self.conv2(x)

        if self.se:
            x = self.se(x)

        if self.drop_path:
            x = self.drop_path(x)

        if self.downsample:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act2(x)
        return x

    def forward(self, x):
        if self.re_parameterized:
            return self.re_parameterized_forward(x)
        return self.train_forward(x)


class ResNet(nn.Module):
    """ResNet / ResNeXt / SE-ResNeXt / SE-Net
    This class implements all variants of ResNet, ResNeXt, SE-ResNeXt, and SENet that
      * have > 1 stride in the 3x3 conv layer of bottleneck
      * have conv-bn-act ordering
    This ResNet impl supports a number of stem and downsample options based on the v1c, v1d, v1e, and v1s
    variants included in the MXNet Gluon ResNetV1b model. The C and D variants are also discussed in the
    'Bag of Tricks' paper: https://arxiv.org/pdf/1812.01187. The B variant is equivalent to torchvision default.
    ResNet variants (the same modifications can be used in SE/ResNeXt models as well):
      * normal, b - 7x7 stem, stem_width = 64, same as torchvision ResNet, NVIDIA ResNet 'v1.5', Gluon v1b
      * c - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64)
      * d - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64), average pool in downsample
      * e - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128), average pool in downsample
      * s - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128)
      * t - 3 layer deep 3x3 stem, stem width = 32 (24, 48, 64), average pool in downsample
      * tn - 3 layer deep 3x3 stem, stem width = 32 (24, 32, 64), average pool in downsample
    ResNeXt
      * normal - 7x7 stem, stem_width = 64, standard cardinality and base widths
      * same c,d, e, s variants as ResNet can be enabled
    SE-ResNeXt
      * normal - 7x7 stem, stem_width = 64
      * same c, d, e, s variants as ResNet can be enabled
    SENet-154 - 3 layer deep 3x3 stem (same as v1c-v1s), stem_width = 64, cardinality=64,
        reduction by 2 on width of first bottleneck convolution, 3x3 downsample convs after first block
    """

    def __init__(
            self,
            block,
            layers,
            num_classes=1000,
            in_chans=3,
            output_stride=32,
            global_pool='avg',
            cardinality=1,
            base_width=64,
            stem_width=64,
            stem_type='',
            replace_stem_pool=False,
            block_reduce_first=1,
            down_kernel_size=1,
            avg_down=False,
            act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d,
            aa_layer=None,
            drop_rate=0.0,
            drop_path_rate=0.,
            drop_block_rate=0.,
            zero_init_last=True,
            block_args=None,
            **kwargs,
    ):
        super(ResNet, self).__init__()
        block_args = block_args or dict()
        assert output_stride in (8, 16, 32)
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.grad_checkpointing = False

        act_layer = get_act_layer(act_layer)
        norm_layer = get_norm_layer(norm_layer)

        # Stem
        deep_stem = 'deep' in stem_type
        inplanes = stem_width * 2 if deep_stem else 64
        if deep_stem:
            stem_chs = (stem_width, stem_width)
            if 'tiered' in stem_type:
                stem_chs = (3 * (stem_width // 4), stem_width)
            self.conv1 = nn.Sequential(*[
                nn.Conv2d(in_chans, stem_chs[0], 3, stride=2, padding=1, bias=False),
                norm_layer(stem_chs[0]),
                act_layer(inplace=True),
                nn.Conv2d(stem_chs[0], stem_chs[1], 3, stride=1, padding=1, bias=False),
                norm_layer(stem_chs[1]),
                act_layer(inplace=True),
                nn.Conv2d(stem_chs[1], inplanes, 3, stride=1, padding=1, bias=False)])
        else:
            if 'cifar' in stem_type:
                self.conv1 = nn.Conv2d(in_chans, inplanes, kernel_size=3, padding=1, bias=False)
            else:
                self.conv1 = nn.Conv2d(in_chans, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(inplanes)
        self.act1 = act_layer(inplace=True)
        self.feature_info = [dict(num_chs=inplanes, reduction=2, module='act1')]

        # Stem pooling. The name 'maxpool' remains for weight compatibility.
        if replace_stem_pool:
            self.maxpool = nn.Sequential(*filter(None, [
                nn.Conv2d(inplanes, inplanes, 3, stride=1 if aa_layer else 2, padding=1, bias=False),
                create_aa(aa_layer, channels=inplanes, stride=2) if aa_layer is not None else None,
                norm_layer(inplanes),
                act_layer(inplace=True)
            ]))
        else:
            if aa_layer is not None:
                if issubclass(aa_layer, nn.AvgPool2d):
                    self.maxpool = aa_layer(2)
                else:
                    self.maxpool = nn.Sequential(*[
                        nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                        aa_layer(channels=inplanes, stride=2)])
            else:
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.Identity() if 'cifar' in stem_type else self.maxpool
        # Feature Blocks
        channels = [64, 128, 256, 512]
        stage_modules, stage_feature_info = make_blocks(
            block,
            channels,
            layers,
            inplanes,
            cardinality=cardinality,
            base_width=base_width,
            output_stride=output_stride,
            reduce_first=block_reduce_first,
            avg_down=avg_down,
            down_kernel_size=down_kernel_size,
            act_layer=act_layer,
            norm_layer=norm_layer,
            aa_layer=aa_layer,
            drop_block_rate=drop_block_rate,
            drop_path_rate=drop_path_rate,
            **dict(block_args, **kwargs)
        )
        for stage in stage_modules:
            self.add_module(*stage)  # layer1, layer2, etc
        self.feature_info.extend(stage_feature_info)

        # Head (Pooling and Classifier)
        self.num_features = 512 * block.expansion
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

        self.init_weights(zero_init_last=zero_init_last)

    @torch.jit.ignore
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

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(stem=r'^conv1|bn1|maxpool', blocks=r'^layer(\d+)' if coarse else r'^layer(\d+)\.(\d+)')
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self, name_only=False):
        return 'fc' if name_only else self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq([self.layer1, self.layer2, self.layer3, self.layer4], x, flatten=True)
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if x.dim() == 5:
            x = x.sum(-1)
        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        return x if pre_logits else self.fc(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _create_resnet(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(ResNet, variant, pretrained, **kwargs)


backbones = {
    'resnet18': dict(block=BasicBlockSub, layers=[2, 2, 2, 2]),
    'resnet34': dict(block=BasicBlockSub, layers=[3, 4, 6, 3]),
    'resnet50': dict(block=BottleneckSub, layers=[3, 4, 6, 3]),
    'resnext50': dict(block=BottleneckSub, layers=[3, 4, 6, 3], cardinality=32, base_width=4),
}
methods = {
    'Sub33': dict(sub_block=SubConvBNBlock, n_block=2),
    'Sub333': dict(sub_block=SubConvBNBlock, n_block=3),
    'NS': dict(sub_block=NS, n_block=4),
}


@register_model
def SubResNet(name, pretrained=False, **kwargs):
    b, m = name.split('_')
    model_args = dict(**backbones[b], **methods[m], **kwargs)
    return _create_resnet('ResNetSub', pretrained, **model_args)


if __name__ == '__main__':
    model = SubResNet('resnet18_NS', stem_type='cifar')
    input = torch.rand(2, 3, 32, 32)
    out = model(input)
