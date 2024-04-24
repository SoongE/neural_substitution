import torch

from src.models import deploy, mobileone
# from src.models.deit import SubDeit
from src.models.mobilenetv1 import AddMobileNet
from src.models.mobilenetv1_sub import SubMobileNet
from src.models.mobileone_sub import mobileoneSub
from src.models.resnet import AddResNet
from src.models.resnet_sub_scriptable import SubResNet
from src.utils.metadata import count_parameters

if __name__ == '__main__':
    # backbones = ['resnet18', 'resnet34', 'resnet50', 'mobilenet', 'mobileone']
    # methods = ['AddInceptionV1', 'AddInceptionV2', 'AddInceptionV3', 'SubInceptionV1', 'SubInceptionV2', 'SubInceptionV3']
    backbones = ['resnet50']
    methods = ['SubV1']

    kwargs = {}
    for b_name in backbones:
        for m_name in methods:
            if b_name.startswith('resnet') and m_name.startswith('Add'):
                model_class = AddResNet
            if b_name.startswith('resnet') and (m_name.startswith('Sub') or m_name.startswith('Hyb')):
                model_class = SubResNet
            if b_name.startswith('mobilenet') and m_name.startswith('Add'):
                model_class = AddMobileNet
            if b_name.startswith('mobilenet') and m_name.startswith('Sub'):
                model_class = SubMobileNet
            if b_name.startswith('mobileone') and m_name.startswith('Add'):
                model_class = mobileone
            if b_name.startswith('mobileone') and m_name.startswith('Sub'):
                model_class = mobileoneSub
            # if b_name.startswith('deit') and m_name.startswith('Sub'):
            #     model_class = SubDeit

            name = f'{b_name}_{m_name}'
            default_model = AddResNet(f'{b_name}_origin')

            default_param = count_parameters(default_model)

            model = model_class(name, neural_drop_rate=0.4, **kwargs)
            model.init_weights(bn_init=True)
            input = torch.rand(12, 3, 224, 224)

            model.eval()
            param = count_parameters(model)
            with torch.no_grad():
                out = model(input)
                deploy(model)
                re_out = model(input)

            if out.dim() == 5:
                out = out.sum(-1)
            print(f"{name} Diff: ", ((out - re_out) ** 2).sum().item())
            re_param = count_parameters(model)

            print(f"Training Parameter: {param / default_param:.3f}")
            print(f"Parameter: {re_param / default_param:.3f}")
