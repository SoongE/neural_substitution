'''MobileNet in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import register_model

from src.models.blocks import SubConvBNBlock, SubInceptionV1Block, SubInceptionV2Block, SubInceptionV3Block, \
    SubInceptionV6Block
from src.models.utils import activation_for_substitute


class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_planes, out_planes, stride=1, block_fn=None, n_block=1, stochastic=None, **kwargs):
        super(Block, self).__init__()
        neural_drop_rate = kwargs.get('neural_drop_rate', 0.0)

        self.conv1 = block_fn(in_planes, in_planes, kernel_size=(3, 3), stride=stride, padding=1, groups=in_planes,
                              n_block=n_block)
        self.conv2 = SubConvBNBlock(in_planes, out_planes, kernel_size=(1, 1), stride=1, padding=0, n_block=n_block,
                                    stochastic=stochastic, neural_drop_rate=neural_drop_rate)

        self.act = nn.ReLU()
        self.re_parameterized = False

    def train_forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(-1)
        xs1 = self.conv1(x)
        x = torch.mean(xs1, dim=4).squeeze(-1)
        x = self.act(x)
        xs1 = activation_for_substitute(xs1, x)

        xs2 = self.conv2(xs1)
        x = torch.mean(xs2, dim=4).squeeze(-1)
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
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.block_args = block_args

        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(1024, num_classes)
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten()

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride, **self.block_args))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)

        if out.dim() == 5:
            out = out.sum(-1)
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
    'SubInceptionV1': dict(block_fn=SubInceptionV1Block, n_block=4),
    'SubInceptionV2': dict(block_fn=SubInceptionV2Block, n_block=3),
    'SubInceptionV3': dict(block_fn=SubInceptionV3Block, n_block=4),
    'SubInceptionV6': dict(block_fn=SubInceptionV6Block, n_block=4),
}


@register_model
def SubMobileNet(name, stochastic=1.0, pretrained=False, **kwargs):
    b, m = name.split('_')
    model_args = dict(**methods[m], stochastic=stochastic, **kwargs)
    return MobileNetV1(**dict(kwargs, **model_args))


if __name__ == '__main__':
    net = SubMobileNet('mob_SubInceptionV6', stem_type='cifar')
    net.eval()
    x = torch.randn(2, 3, 32, 32)
    y = net(x)

    # deploy(net, Block)
    # yy = net(x)

    # print(((y - yy) ** 2).sum().item())
