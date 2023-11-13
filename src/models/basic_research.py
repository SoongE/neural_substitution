import time

import torch

from src.models import deploy, mobileone
from src.models.deit import SubDeit
from src.models.mobilenetv1 import AddMobileNet
from src.models.mobilenetv1_sub import SubMobileNet
from src.models.resnet import AddResNet
from src.models.resnet_sub import SubResNet

if __name__ == '__main__':
    # backbones = ['resnet18', 'resnet34', 'resnet50', 'mobilenet', 'mobileone']
    # methods = ['AddInceptionV1', 'AddInceptionV2', 'AddInceptionV3', 'SubInceptionV1', 'SubInceptionV2', 'SubInceptionV3']
    backbones = ['deitTiny']
    methods = ['SubMlp3']

    kwargs = {}
    for b_name in backbones:
        for m_name in methods:
            if b_name.startswith('resnet') and m_name.startswith('Add'):
                model_class = AddResNet
            if b_name.startswith('resnet') and m_name.startswith('Sub'):
                model_class = SubResNet
            if b_name.startswith('mobilenet') and m_name.startswith('Add'):
                model_class = AddMobileNet
            if b_name.startswith('mobilenet') and m_name.startswith('Sub'):
                model_class = SubMobileNet
            if b_name.startswith('mobileone') and m_name.startswith('Add'):
                model_class = mobileone
                kwargs.update({'substitution': False})
            if b_name.startswith('mobileone') and m_name.startswith('Sub'):
                model_class = mobileone
                kwargs.update({'substitution': True})
            if b_name.startswith('deit') and m_name.startswith('Sub'):
                model_class = SubDeit

            name = f'{b_name}_{m_name}'

            model = model_class(name, stochastic=1.0, neural_drop_rate=0.2, **kwargs)
            model.init_weights(bn_init=True)
            input = torch.rand(12, 3, 224, 224)
            s = time.time()
            model.eval()
            with torch.no_grad():
                out = model(input)
                deploy(model)
                re_out = model(input)
            print(time.time() - s)
            if out.dim() == 5:
                out = out.sum(-1)
            print(f"{name}: ", ((out - re_out) ** 2).sum().item())
