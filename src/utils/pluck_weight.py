import os
import shutil
from pathlib import Path
from pprint import pprint

ids = {
    ### CIFAR
    # 'mobilenetv1_SubInceptionV3': '90s8q1au',
    # 'mobilenetv1_SubInceptionV2': '9miont50',
    # 'mobilenetv1_SubInceptionV1': 'fxv3b7uh',
    # 'mobilenetv1_AddInceptionV3': 'zfgj9v0l',
    # 'mobilenetv1_AddInceptionV2': 'xr7jp082',
    # 'mobilenetv1_AddInceptionV1': 'd15kin7v',
    # 'mobilenetv1': 'jrqmq1hm',

    # 'resnet18_SubInceptionV3': 'pemmqz0y',
    # 'resnet18_SubInceptionV2': '5so49l26',
    # 'resnet18_SubInceptionV1': 'z20mxcy4',
    # 'resnet18_AddInceptionV3': 'zp9gcykm',
    # 'resnet18_AddInceptionV2': 'bl2s7jw8',
    # 'resnet18_AddInceptionV1': 'tuntzznx',
    # 'resnet18': 'zjjtxb8c',

    # 'resnet34_SubInceptionV3': 'p5cdeyuj',
    # 'resnet34_SubInceptionV2': 'kda0d0o9',
    # 'resnet34_SubInceptionV1': 'xfcgi8xp',
    # 'resnet34_AddInceptionV3': 's603vnw0',
    # 'resnet34_AddInceptionV2': '113uwx13',
    # 'resnet34_AddInceptionV1': 'im4gczkq',
    # 'resnet34': 'w5k8ejzs'

    # 'resnext50_SubInceptionV3': 'zw3oxqjo',
    # 'resnext50_SubInceptionV2': 'zfjsgy3l',
    # 'resnext50_SubInceptionV1': 'zem3bbg0',
    # 'resnext50_AddInceptionV3': 'v8gojoi2',
    # 'resnext50_AddInceptionV2': 'virgotgu',
    # 'resnext50_AddInceptionV1': 'wgmz2qxi',
    # 'resnext50': 'xdfmatnu',

    # 'resnet50_SubInceptionV3': '1qump7q5',
    # 'resnet50_SubInceptionV2': '22zgfbd5',
    # 'resnet50_SubInceptionV1': '016vu5ax',
    # 'resnet50_AddInceptionV3': 'oro9tms0',
    # 'resnet50_AddInceptionV2': 'cwh958e',
    # 'resnet50_AddInceptionV1': '4ki1cy6d',
    # 'resnet50': 't5hvblsu',


    'resnet18_Sub33333': 'o4u1zw29',
    'resnet18_Sub3333': 'd1fz5yqq',
    'resnet18_Sub333': 'cea3apn3',
    'resnet18_Sub33': 'so6jh96v',
    # 'resnet18_Add6': '59i1tjji',
    # 'resnet18_Add33333': 'fbozwpgl',
    # 'resnet18_Add3333': '31rhwru1',
    # 'resnet18_Add333': 'cv0qx6aw',
    # 'resnet18_Add33': 'ajpcyt3e',

    ### ImageNet
    # 'mobilenetv1_SubInceptionV3': '401xzd37',
    # 'mobilenetv1_SubInceptionV2': 'i2b1stc5',
    # 'mobilenetv1_SubInceptionV1': 'n2uc8yd5',
    # 'mobilenetv1_AddInceptionV3': 'jc6pdvc4',
    # 'mobilenetv1_AddInceptionV2': '8l6au804',
    # 'mobilenetv1_AddInceptionV1': 'p3ngec5t',
    # 'mobilenetv1': 'dkxksuhn',

    # 'resnet50_SubInceptionV3': 'wqlpdkzh',
    # 'resnet50_SubInceptionV2': '3wa7is6c',
    # 'resnet50_SubInceptionV1': '37wlwktb',
    # 'resnet50_AddInceptionV3': 'r25bdy3j',
    # 'resnet50_AddInceptionV2': 'hxmyamnt',
    # 'resnet50_AddInceptionV1': 'lsxp2bbd',
}

save_weight = Path('/home/seungmin/dmount/neural_substitution/weights/cifar/multi')
save_weight.mkdir(parents=True, exist_ok=True)

fail_list = []
success_list = []


def pluck(multirun):
    if multirun:
        root = Path('/home/seungmin/dmount/neural_substitution/multiruns')
    else:
        root = Path('/home/seungmin/dmount/neural_substitution/runs')
    for k, v in ids.items():
        copy = False
        files = root.rglob(f'*{k}*/*/*/*/*' if multirun else f'*{k}*/*/*/*')
        for file in files:
            print(file)
            if v in str(file):
                print(v)
                source = list(file.parents)[1]
                target = save_weight / 'source' / f'{k}_{v}'
                model = target.parent.name + '/' + target.name + '/model_best.pth.tar'
                shutil.copytree(source, target)
                os.symlink(model, save_weight / f'{k}.pth.tar')
                print(k, v, model, source)
                copy = True
                break
        if copy:
            success_list.append(f'{k} {v}')
        else:
            fail_list.append(f'{k} {v}')


# pluck(multirun=True)
#
# for s_item in success_list:
#     ids.pop(s_item.split()[0])
#
pluck(multirun=False)
# pluck(multirun=True)

print('SUCCESS: ')
pprint(success_list)

print('FAIL:')
pprint(set(fail_list))
