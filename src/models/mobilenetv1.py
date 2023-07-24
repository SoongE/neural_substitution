'''MobileNet in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import register_model

from src.models.blocks import AddInceptionV1Block, AddInceptionV3Block, ConvBNBlock, AddConvBNBlock, AddInceptionV2Block


class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_planes, out_planes, stride=1, block_fn=None, **kwargs):
        super(Block, self).__init__()
        self.conv1 = block_fn(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, n_block=kwargs['n_block'],
                              groups=in_planes)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_planes)
        )
        self.act = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()

    def forward(self, x):
        out = self.act(self.conv1(x))
        out = self.act(self.conv2(out))
        return out


class MobileNetV1(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]

    def __init__(self, num_classes=100, block_fn=None, **kwargs):
        super(MobileNetV1, self).__init__()
        stem_type = kwargs.pop('stem_type', '')
        if 'cifar' in stem_type:
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.block_fn = block_fn
        self.layers = self._make_layers(in_planes=32, **kwargs)
        self.linear = nn.Linear(1024, num_classes)
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten()

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

    def _make_layers(self, in_planes, **kwargs):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride, self.block_fn, **kwargs))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)

        out = self.pool(out)
        out = self.flatten(out)
        out = self.linear(out)
        return out


methods = {
    '': dict(block_fn=ConvBNBlock, n_block=1),
    'Add33': dict(block_fn=AddConvBNBlock, n_block=2),
    'Add333': dict(block_fn=AddConvBNBlock, n_block=3),
    'AddInceptionV1': dict(block_fn=AddInceptionV1Block, n_block=4),
    'AddInceptionV2': dict(block_fn=AddInceptionV2Block, n_block=3),
    'AddInceptionV3': dict(block_fn=AddInceptionV3Block, n_block=4),
}


@register_model
def AddMobileNet(name, pretrained=False, **kwargs):
    name = name.split('_')
    if len(name) == 1:
        b, m = name, ''
    else:
        b, m = name
    return MobileNetV1(**dict(kwargs, **methods[m]))


if __name__ == '__main__':
    import torch

    net = AddMobileNet('mobilent_Add33')
    net.eval()
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
