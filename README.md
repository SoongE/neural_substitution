# Neural Substitution

Official PyTorch implementation for Neural Substitution. For details, see the
paper: [Neural Substitution for Branch-level Network Re-parameterization](TBD)

## How to run

### Requirements

Please install the requirements including pytorch for stable running. This code has been developed with python 3.10,
PyTorch 2.2.1, and CUDA 12.1.

```bash
pip install -r requirements.txt
```

### Training

We use 4 GPU for training. You can modify the model_name as [`resnet18_NS` | `resnet50_NS`]

```bash
torchrun --nproc_per_node=4 main.py --config-name=imagenet gpus=[0,1,2,3] train.batch_size=64 train.optimizer.grad_accumulation=4 model.model_name=resnet50_NS
```

### Validate Re-parameterization

To validate the re-parameterization, run the code below. This will display the accuracy before and after the
re-parameterization.

```bash
python val_reparam.py
```

## Folder structure of dataset

The dataset should exist in a folder called `data`, in the form shown below, with the same folder name.

```
data
├── cifar-100-python
│   ├── file.txt~
│   ├── meta
│   ├── test
│   └── train
└── imageNet
    ├── train
    ├── val
    └── meta.bin
```

## License

This repository is released under the Apache 2.0 license as found in the [LICENSE](./LICENSE) file.

## Citation

If our project is helpful for your research, please consider citation:paperclip: and giving a star:star2: :

```
TBD
```
