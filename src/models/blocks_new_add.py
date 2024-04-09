from torch import nn

from src.models.blocks_new import SubConvBNBlock, SubV1, SubV2, SubV3, SubV4

class ConvBNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_block, stride=1, padding=0, bias=False,
                 bn=nn.BatchNorm2d, groups=1, neural_drop_rate=0.0, **kwargs):
        super().__init__()
        self.convbn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, groups=groups, **kwargs),
            bn(out_channels),
        )

    def forward(self, x):
        return self.convbn(x)

class AddConvBNBlock(SubConvBNBlock):
    def forward(self, x):
        if self._is_deploy:
            return self.deploy_forward(x)

        out = 0
        for _, block in self.blocks.items():
            out = out + block(x)
        return out


class AddV1(SubV1):
    def forward(self, x):
        if self._is_deploy:
            return self.deploy_forward(x)

        out = 0
        for _, block in self.blocks.items():
            out = out + block(x)
        return out


class AddV2(SubV2):
    def forward(self, x):
        if self._is_deploy:
            return self.deploy_forward(x)

        out = 0
        for _, block in self.blocks.items():
            out = out + block(x)
        return out


class AddV3(SubV3):
    def forward(self, x):
        if self._is_deploy:
            return self.deploy_forward(x)

        out = 0
        for _, block in self.blocks.items():
            out = out + block(x)
        return out


class AddV4(SubV4):
    def forward(self, x):
        if self._is_deploy:
            return self.deploy_forward(x)

        out = 0
        for _, block in self.blocks.items():
            out = out + block(x)
        return out
