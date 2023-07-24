f# Neural Substitution

### Experiment

- ResNet50 (75.45)
- ResNet50_sub2Shuffle (75.83)
- ResNet18
- ResNet18_sub2Shuffle

```bash
torchrun --nproc_per_node=4 main.py gpus=[0,1,2,3] model.model_name=resnet18_Sub311 wandb=True name=resnet18_Sub311
```