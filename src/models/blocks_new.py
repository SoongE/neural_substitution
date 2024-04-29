from abc import abstractmethod

import torch
from torch import nn

from src.models.reparam_utils import get_equivalent_kernel_bias, merge_1x1_kxk, fuse_bn, avg_to_kernel, expend_kernel, \
    BNAndPadLayer


def drop_path_topology(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], x.shape[1]) + (1,) * (x.ndim - 2)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class _SubstituteABC(nn.Module):
    blocks: nn.ModuleDict
    deploy_blocks: nn.Module
    n_flow: int = 1
    neural_drop_rate: float = .0
    _is_deploy: bool = False

    def substitute(self, x: torch.Tensor):
        n_batch = x.size(0)
        n_x = x.size(-1)
        n_conv = len(self.blocks)

        x_out = [0] * n_x
        if self.training:
            rand_idx = torch.randperm(n_conv * n_x) % n_x
        else:
            rand_idx = torch.arange(n_conv * n_x) % n_x
        p_idx = 0

        x = x.permute(4, 0, 1, 2, 3).flatten(0, 1)
        for _, conv in self.blocks.items():
            _out = drop_path_topology(conv(x).unflatten(0, (n_x, n_batch)), self.neural_drop_rate, self.training)
            for i, idx in enumerate(rand_idx[p_idx * n_x: (p_idx + 1) * n_x]):
                x_out[idx] = x_out[idx] + _out[i]
            p_idx = p_idx + 1

        x_out = torch.stack(x_out, dim=0)
        return x_out.permute(1, 2, 3, 4, 0)

    def forward(self, x):
        if self._is_deploy:
            return self.deploy_forward(x)
        self.n_flow = x.size(-1)
        return self.substitute(x)

    def deploy_forward(self, x):
        return self.deploy_blocks(x)

    @abstractmethod
    def re_parameterization(self):
        """
        Re-parameterize all blocks to single block for deploy.
        """
        # Do re-parameterize
        self.deploy_blocks = nn.Identity()
        self._is_deploy = True


class SubConvBNBlock(_SubstituteABC):
    def __init__(self, in_channels, out_channels, kernel_size, n_block, stride=1, padding=0, bias=False,
                 bn=nn.BatchNorm2d, groups=1, neural_drop_rate=0.0):
        super().__init__()
        self.args = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'bias': bias,
            'groups': groups,
        }
        self.n_block = n_block
        self.neural_drop_rate = neural_drop_rate

        self.blocks = nn.ModuleDict()
        for i in range(n_block):
            self.blocks.update({f'kxk_{i}': nn.Sequential(
                nn.Conv2d(**self.args),
                bn(out_channels),
            )})

    def re_parameterization(self):
        self.args['bias'] = True
        self.deploy_blocks = nn.Conv2d(**self.args, device=self.blocks['kxk_0'][0].weight.device)

        eq_k, eq_b = get_equivalent_kernel_bias(self.blocks, self.n_flow)

        self.deploy_blocks.weight.data = eq_k
        self.deploy_blocks.bias.data = eq_b

        self.__delattr__('blocks')
        self._is_deploy = True


class SubStem(_SubstituteABC):
    def __init__(self, in_channels, out_channels, kernel_size, n_block=4, stride=1, padding=0, bias=False,
                 bn=nn.BatchNorm2d, groups=1, neural_drop_rate=0.0):
        super().__init__()
        assert kernel_size == 7
        self.args = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            # 'stride': stride,
            # 'padding': padding,
            'bias': bias,
            'groups': groups,
        }
        self.n_block = n_block
        self.neural_drop_rate = neural_drop_rate

        self.blocks = nn.ModuleDict()

        self.blocks.update({'1x1': nn.Sequential(
            nn.Conv2d(**self.args, kernel_size=1, stride=2, padding=0),
            bn(out_channels),
        )})

        self.blocks.update({'3x3': nn.Sequential(
            nn.Conv2d(**self.args, kernel_size=3, stride=2, padding=1),
            bn(out_channels),
        )})

        self.blocks.update({'5x5': nn.Sequential(
            nn.Conv2d(**self.args, kernel_size=5, stride=2, padding=2),
            bn(out_channels),
        )})

        self.blocks.update({'7x7': nn.Sequential(
            nn.Conv2d(**self.args, kernel_size=7, stride=2, padding=3),
            bn(out_channels),
        )})

    def re_parameterization(self):
        self.args['bias'] = True
        self.deploy_blocks = nn.Conv2d(**self.args, kernel_size=7, stride=2, padding=3, device=self.blocks['1x1'][0].weight.device)

        eq_k, eq_b = 0, 0
        for key, value in self.blocks.items():
            k, b = fuse_bn(value[0], value[1], self.n_flow)
            if k != '7x7': k = expend_kernel(k, 7)
            eq_k += k
            eq_b += b

        self.deploy_blocks.weight.data = eq_k
        self.deploy_blocks.bias.data = eq_b

        self.__delattr__('blocks')
        self._is_deploy = True


class SubV1(_SubstituteABC):  # DBB
    def __init__(self, in_channels, out_channels, kernel_size, n_block=4, stride=1, padding=0, bias=False,
                 bn=nn.BatchNorm2d, groups=1, neural_drop_rate=0.0, **kwargs):
        super().__init__()
        assert n_block == 4
        self.args = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'bias': bias,
            'groups': groups,
        }
        self.n_block = n_block
        self.neural_drop_rate = neural_drop_rate

        self.blocks = nn.ModuleDict()

        # kxk
        self.blocks.update({'kxk': nn.Sequential(
            nn.Conv2d(**self.args),
            bn(out_channels),
        )})

        # 1x1
        self.blocks.update({'1x1': nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride, bias=False, **kwargs),
            bn(out_channels),
        )})

        # 1x1-kxk
        self.blocks.update({'1x1-kxk': nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1), bias=False, groups=groups),
            BNAndPadLayer(padding, in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False, **kwargs),
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
        self.args['bias'] = True
        self.deploy_blocks = nn.Conv2d(**self.args, device=self.blocks['kxk'][0].weight.device)

        _k0, _b0 = fuse_bn(*self.blocks['kxk'], self.n_flow)

        _k1, _b1 = fuse_bn(*self.blocks['1x1'], self.n_flow)
        _k1 = expend_kernel(_k1, self.args['kernel_size'])

        _k2, _b2 = fuse_bn(*self.blocks['1x1-kxk'][:2], self.n_flow)
        _k22, _b22 = fuse_bn(*self.blocks['1x1-kxk'][2:], self.n_flow)
        _k2, _b2 = merge_1x1_kxk(_k2, _b2, _k22, _b22, self.deploy_blocks.groups)

        _k3, _b3 = fuse_bn(*self.blocks['1x1-avg'][:2], self.n_flow)
        _k33 = avg_to_kernel(self.deploy_blocks.out_channels, self.deploy_blocks.kernel_size, self.deploy_blocks.groups)
        _k33, _b33 = fuse_bn(_k33.to(self.blocks['1x1-avg'][0].weight.device), self.blocks['1x1-avg'][3], self.n_flow)
        _k3, _b3 = merge_1x1_kxk(_k3, _b3, _k33, _b33, self.deploy_blocks.groups)

        self.deploy_blocks.weight.data = sum([_k0, _k1, _k2, _k3])
        self.deploy_blocks.bias.data = sum([_b0, _b1, _b2, _b3])
        self.__delattr__('blocks')
        self._is_deploy = True


class SubV2(_SubstituteABC):  # ACNet
    def __init__(self, in_channels, out_channels, kernel_size, n_block=3, stride=1, padding=0, bias=False,
                 bn=nn.BatchNorm2d, groups=1, neural_drop_rate=0.0, **kwargs):
        super().__init__()
        assert n_block == 3
        self.args = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'bias': bias,
            'groups': groups,
        }
        self.n_block = n_block
        self.neural_drop_rate = neural_drop_rate

        self.blocks = nn.ModuleDict()

        # kxk
        self.blocks.update({'kxk': nn.Sequential(
            nn.Conv2d(**self.args),
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
        self.args['bias'] = True
        self.deploy_blocks = nn.Conv2d(**self.args)

        _k0, _b0 = fuse_bn(*self.blocks['kxk'], self.n_flow)

        _k1, _b1 = fuse_bn(*self.blocks['1x3'], self.n_flow)
        _k1 = expend_kernel(_k1, self.args['kernel_size'])

        _k2, _b2 = fuse_bn(*self.blocks['3x1'], self.n_flow)
        _k2 = expend_kernel(_k2, self.args['kernel_size'])

        self.deploy_blocks.weight.data = sum([_k0, _k1, _k2])
        self.deploy_blocks.bias.data = sum([_b0, _b1, _b2])
        self.__delattr__('blocks')
        self._is_deploy = True


class SubV3(_SubstituteABC):
    def __init__(self, in_channels, out_channels, kernel_size, n_block=4, stride=1, padding=0, bias=False,
                 bn=nn.BatchNorm2d, groups=1, neural_drop_rate=0.0, **kwargs):
        super().__init__()
        assert n_block == 4
        self.args = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'bias': bias,
            'groups': groups,
        }
        self.n_block = n_block
        self.neural_drop_rate = neural_drop_rate

        self.blocks = nn.ModuleDict()

        # kxk
        self.blocks.update({'kxk': nn.Sequential(
            nn.Conv2d(**self.args),
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
        self.args['bias'] = True
        self.deploy_blocks = nn.Conv2d(**self.args)

        _k0, _b0 = fuse_bn(*self.blocks['kxk'], self.n_flow)

        _k1, _b1 = fuse_bn(*self.blocks['1x1'], self.n_flow)
        _k1 = expend_kernel(_k1, self.args['kernel_size'])

        _k2, _b2 = fuse_bn(*self.blocks['1x3'], self.n_flow)
        _k2 = expend_kernel(_k2, self.args['kernel_size'])

        _k3, _b3 = fuse_bn(*self.blocks['3x1'], self.n_flow)
        _k3 = expend_kernel(_k3, self.args['kernel_size'])

        self.deploy_blocks.weight.data = sum([_k0, _k1, _k2, _k3])
        self.deploy_blocks.bias.data = sum([_b0, _b1, _b2, _b3])
        self.__delattr__('blocks')
        self._is_deploy = True


class SubV4(_SubstituteABC):
    def __init__(self, in_channels, out_channels, kernel_size, n_block=4, stride=1, padding=0, bias=False,
                 bn=nn.BatchNorm2d, groups=1, neural_drop_rate=0.0, hidden_ratio=2, **kwargs):
        super().__init__()
        assert n_block == 4
        self.args = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'bias': bias,
            'groups': groups,
        }
        self.n_block = n_block
        self.neural_drop_rate = neural_drop_rate
        hidden_channels = int(in_channels * hidden_ratio)

        self.blocks = nn.ModuleDict()

        self.blocks.update({'kxk': nn.Sequential(
            nn.Conv2d(**self.args),
            bn(out_channels),
        )})

        self.blocks.update({'1x1': nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride, bias=False, groups=groups,
                      **kwargs),
            bn(out_channels),
        )})

        self.downsample = (in_channels != out_channels)
        if self.downsample:
            self.blocks.update({'dsx1': nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=1, bias=False, groups=groups, **kwargs),
                BNAndPadLayer(padding, out_channels),
                nn.AvgPool2d(kernel_size=kernel_size, stride=stride),
            )})
        else:
            self.blocks.update({'dsx1': nn.Sequential(
                BNAndPadLayer(padding, out_channels),
                nn.AvgPool2d(kernel_size=kernel_size, stride=stride),
            )})

        self.blocks.update({'dsx2': nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=(1, 1), bias=False, groups=groups),
            BNAndPadLayer(padding, hidden_channels),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False, groups=groups),
            bn(out_channels),
        )})

    def re_parameterization(self):
        self.args['bias'] = True
        self.deploy_blocks = nn.Conv2d(**self.args, device=self.blocks['kxk'][0].weight.device)

        _k0, _b0 = fuse_bn(*self.blocks['kxk'], self.n_flow)

        _k1, _b1 = fuse_bn(*self.blocks['1x1'], self.n_flow)
        _k1 = expend_kernel(_k1, self.args['kernel_size'])

        if self.downsample:
            _k2, _b2 = fuse_bn(*self.blocks['dsx1'][:2], self.n_flow)
            _k22 = avg_to_kernel(self.deploy_blocks.out_channels, self.deploy_blocks.kernel_size,
                                 self.deploy_blocks.groups).to(self.blocks['dsx1'][0].weight.device)
            _k2, _b2 = merge_1x1_kxk(_k2, _b2, _k22, 0, self.deploy_blocks.groups)
        else:
            _k22 = avg_to_kernel(self.deploy_blocks.out_channels, self.deploy_blocks.kernel_size,
                                 self.deploy_blocks.groups)
            _k2, _b2 = fuse_bn(_k22.to(self.deploy_blocks.weight.device), self.blocks['dsx1'][0], self.n_flow)

        _k3, _b3 = fuse_bn(*self.blocks['dsx2'][:2], self.n_flow)
        _k33, _b33 = fuse_bn(*self.blocks['dsx2'][2:], self.n_flow)
        _k3, _b3 = merge_1x1_kxk(_k3, _b3, _k33, _b33, self.args.get('groups', 1))

        self.deploy_blocks.weight.data = sum([_k0, _k1, _k2, _k3])
        self.deploy_blocks.bias.data = sum([_b0, _b1, _b2, _b3])
        self.__delattr__('blocks')
        self._is_deploy = True


if __name__ == '__main__':
    n_block = 4
    conv = SubStem(3, 3, 7, stride=2, padding=3, neural_drop_rate=0.4)
    conv.eval()

    x = torch.rand(2, 3, 224, 224, n_block)

    out = conv(x)
    conv.re_parameterization()
    re_out = conv(x.sum(-1))
    print(f'Diff: {((out.sum(-1) - re_out) ** 2).sum().item()}')
