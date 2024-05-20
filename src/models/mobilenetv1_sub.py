'''MobileNet in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
from timm.models import register_model

from src.models.blocks_new import SubConvBNBlock, SubV1, SubV2, SubV3, SubV4, SubStem, SubV7
from src.models.blocks_new_add import AddConvBNBlockOne
from src.models.utils import activation_for_substitute


class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_planes, out_planes, stride=1, **kwargs):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_planes)
        )
        self.act = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()

    def forward(self, x):
        if x.dim() == 5:
            x = x.sum(-1)
        out = self.act(self.conv1(x))
        out = self.act(self.conv2(out))
        return out


class SubBlock(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_planes, out_planes, stride=1, block_fn=None, n_block=4, **kwargs):
        super(SubBlock, self).__init__()
        neural_drop_rate = kwargs.get('neural_drop_rate', 0.0)
        self.conv1 = block_fn(in_planes, in_planes, kernel_size=(3, 3), stride=stride, padding=1, groups=in_planes,
                              n_block=n_block, neural_drop_rate=neural_drop_rate)
        self.conv2 = AddConvBNBlockOne(in_planes, out_planes, kernel_size=(1, 1), stride=1, padding=0, n_block=4)

        self.act = nn.ReLU()
        self.re_parameterized = False

    def train_forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(-1)
        xs1 = self.conv1(x)
        x = torch.sum(xs1, dim=4).squeeze(-1)
        x = self.act(x)
        xs1 = activation_for_substitute(xs1, x)

        xs2 = self.conv2(xs1)
        x = torch.sum(xs2, dim=4).squeeze(-1)
        x = self.act(x)
        xs2 = activation_for_substitute(xs2, x)

        return xs2

    def re_parameterized_forward(self, x):
        out = self.act(self.conv1(x))
        out = self.act(self.conv2(out))
        return out

    def forward(self, x):
        if self.re_parameterized:
            return self.re_parameterized_forward(x)
        return self.train_forward(x)


class MobileNetV1(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]

    def __init__(self, num_classes=100, **block_args):
        super(MobileNetV1, self).__init__()
        stem_type = block_args.pop('stem_type', '')
        if 'cifar' in stem_type:
            self.conv1 = SubV1(3, 32, kernel_size=3, n_block=4, padding=1, bias=False)
        else:
            self.conv1 = SubStem(3, 32, kernel_size=7, stride=2, padding=3, bias=False)

        self.n_block = 4
        self.block_args = block_args
        self.act = nn.ReLU(inplace=True)

        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(1024, num_classes)
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten()

        self.forward = self.forward_train

    def re_parameterization(self):
        self.forward = self.forward_deploy

    def _make_layers(self, in_planes):
        layers = []
        for i, x in enumerate(self.cfg):
            block_fn = SubBlock if i < 6 else Block
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(block_fn(in_planes, out_planes, stride, **self.block_args))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward_train(self, x):
        xs = (x / self.n_block).unsqueeze(-1).repeat(1, 1, 1, 1, self.n_block)
        xs = self.conv1(xs)
        x = self.act(torch.sum(xs, dim=4).squeeze(-1))
        xs = activation_for_substitute(xs, x)

        out = self.layers(xs)
        if out.dim() == 5:
            out = out.sum(-1)
        out = self.pool(out)
        out = self.flatten(out)
        out = self.linear(out)
        return out

    def forward_deploy(self, x):
        out = self.act(self.conv1(x))
        out = self.layers(out)

        out = self.pool(out)
        out = self.flatten(out)
        out = self.linear(out)
        return out

    @torch.jit.ignore
    def init_weights(self, bn_init=False):
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if bn_init and isinstance(m, nn.BatchNorm2d):
                nn.init.uniform_(m.running_mean, 0, 0.1)
                nn.init.uniform_(m.running_var, 0, 0.1)
                nn.init.uniform_(m.weight, 0, 0.1)
                nn.init.uniform_(m.bias, 0, 0.1)


backbones = {
}
methods = {
    'Sub33': dict(block_fn=SubConvBNBlock, n_block=2),
    'Sub333': dict(block_fn=SubConvBNBlock, n_block=3),
    'StemV1C': dict(block_fn=SubV1, n_block=4),
    'StemV2C': dict(block_fn=SubV2, n_block=3),
    'StemV3C': dict(block_fn=SubV3, n_block=4),
    'StemV4C': dict(block_fn=SubV4, n_block=4),
    'StemV7C': dict(block_fn=SubV7, n_block=3),
}


@register_model
def SubMobileNet(name, pretrained=False, **kwargs):
    b, m = name.split('_')
    dpr = kwargs.get('drop_path_rate', 0.)
    if dpr != 0:
        kwargs.update({'drop_path_rate': 0., 'neural_drop_rate': dpr * 0.001})
    model_args = dict(**methods[m], **kwargs)
    return MobileNetV1(**dict(kwargs, **model_args))


if __name__ == '__main__':
    net = SubMobileNet('mob_StemV1C', stem_type='cifar', drop_path_rate=0.1)
    net.eval()
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
