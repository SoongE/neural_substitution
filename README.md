# Neural Substitution

All weights are saved in Docker8 *PROTECTED/NeuralSubstitution_Weights*

```bash
torchrun --nproc_per_node=4 main.py gpus=[0,1,2,3] model.model_name=resnet18_Sub311 wandb=True name=resnet18_Sub311
```