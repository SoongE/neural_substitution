import pandas
import wandb

wandb.init(project='Neural_Substitution_Cifar100', entity='soonge', name='resnet18_AddInceptionV3',
           settings=wandb.Settings(_disable_stats=True))

csv = pandas.read_csv(
    '/home/seungmin/dmount/neural_substitution/runs/cifar100_resnet18_AddInceptionV3/resnet18_AddInceptionV3_20230714-132552/summary.csv',
    names=['epoch', 'train_loss', 'eval_loss', 'eval_top1', 'eval_top5', 'eval_Best_Top1'], header=None)

for i, row in csv.iterrows():
    row = dict(row)
    for k, v in row.items():
        if isinstance(v, int):
            continue
        row[k] = float(v.split('tensor(')[-1].split(', ')[0])

    wandb.log(row)
