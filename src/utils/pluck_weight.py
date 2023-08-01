import shutil
from pathlib import Path
from pprint import pprint

ids = {
    'resnet34_SubInceptionV3': 'p5cdeyuj',
    'resnet34_SubInceptionV2': 'kda0d0o9',
    'resnet34_SubInceptionV1': 'xfcgi8xp',
    'resnet34_AddInceptionV3': 's603vnw0',
    'resnet34_AddInceptionV2': '113uwx13',
    'resnet34_AddInceptionV1': 'im4gczkq',
    'resnet34': 'w5k8ejzs'
}

save_weight = Path('/home/seungmin/dmount/neural_substitution-backup/weights/resnet34')
save_weight.mkdir(parents=True, exist_ok=True)

root = Path('/home/seungmin/dmount/neural_substitution-backup/runs')
files = root.rglob('*')

fail_list = []
success_list = []
for k, v in ids.items():
    copy = False
    files = root.rglob(f'*{k}*/*/*')
    for file in files:
        if v in str(file):
            model = file.parent.parent / 'model_best.pth.tar'
            shutil.copy(model, save_weight / f'{k}.pth.tar')
            print(k, v, model)
            copy = True

    if copy:
        success_list.append(f'{k} {v}')
    else:
        fail_list.append(f'{k} {v}')

print('SUCCESS: ')
pprint(success_list)

print('FAIL:')
pprint(fail_list)
