import math
import random
from functools import partial
from math import floor

import torch
import torch.nn.functional as F
from timm.data.loader import _worker_init
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.dataloader import default_collate, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100, CIFAR10
from torchvision.datasets.samplers import DistributedSampler
from torchvision.transforms import RandomChoice


class TrainTransform:
    def __init__(self, resize, resize_mode, pad, scale, ratio, hflip, auto_aug, remode, interpolation, mean, std):
        interpolation = transforms.functional.InterpolationMode(interpolation)

        transform_list = []

        if hflip:
            transform_list.append(transforms.RandomHorizontalFlip(hflip))

        if auto_aug:
            if auto_aug.startswith('ra'):
                transform_list.append(transforms.RandAugment(interpolation=interpolation))
            elif auto_aug.startswith('ta_wide'):
                transform_list.append(transforms.TrivialAugmentWide(interpolation=interpolation))
            elif auto_aug.startswith('aa'):
                policy = transforms.AutoAugmentPolicy('cifar10')
                transform_list.append(transforms.AutoAugment(policy=policy, interpolation=interpolation))

        if resize_mode == 'RandomResizedCrop':
            transform_list.append(
                transforms.RandomResizedCrop(resize, scale=scale, ratio=ratio, interpolation=interpolation))
        elif resize_mode == 'ResizeRandomCrop':
            transform_list.extend([transforms.Resize(resize, interpolation=interpolation),
                                   transforms.RandomCrop(resize, padding=pad)])
        else:
            assert f"{resize_mode} should be RandomResizedCrop and ResizeRandomCrop"

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        if remode:
            transform_list.append(transforms.RandomErasing(remode))

        self.transform_fn = transforms.Compose(transform_list)

    def __call__(self, x):
        return self.transform_fn(x)


class ValTransform:
    def __init__(self, size, resize_mode, crop_ptr, interpolation, mean, std):
        interpolation = transforms.functional.InterpolationMode(interpolation)

        if not isinstance(size, (tuple, list)):
            size = (size, size)

        resize = (int(floor(size[0] / crop_ptr)), int(floor(size[1] / crop_ptr)))

        if resize_mode == 'resize_shorter':
            resize = resize[0]

        transform_list = [
            transforms.Resize(resize, interpolation=interpolation),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]

        self.transform_fn = transforms.Compose(transform_list)

    def __call__(self, x):
        return self.transform_fn(x)


class MixUP:
    def __init__(self, p=0.5, alpha=1.0, nclass=1000):
        self.p = p
        self.alpha = alpha
        self.nclass = nclass

    def __call__(self, batch, target):
        if self.p > random.random():
            return batch, target
        if target.ndim == 1:
            target = F.one_hot(target, num_classes=self.nclass).to(dtype=batch.dtype)

        ratio = float(1 - torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])

        batch_roll = batch.roll(1, 0)
        target_roll = target.roll(1, 0)

        batch = batch * (1 - ratio) + batch_roll * ratio
        target = target * (1 - ratio) + target_roll * ratio

        return batch, target


class CutMix:
    def __init__(self, p=0.5, alpha=1.0, nclass=1000):
        self.p = p
        self.alpha = alpha
        self.nclass = nclass

    @torch.inference_mode()
    def __call__(self, batch, target):
        if self.p > random.random():
            return batch, target

        if target.ndim == 1:
            target = F.one_hot(target, num_classes=self.nclass).to(dtype=batch.dtype)

        B, C, H, W = batch.shape
        ratio = float(1 - torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])

        batch_roll = batch.roll(1, 0)
        target_roll = target.roll(1, 0)

        height_half = int(0.5 * math.sqrt(ratio) * H)
        width_half = int(0.5 * math.sqrt(ratio) * W)
        r = int(random.random() * H)
        c = int(random.random() * W)

        start_x = max(r - height_half, 0)
        end_x = min(r + height_half, H)
        start_y = max(r - width_half, 0)
        end_y = min(r + width_half, W)

        ratio = 1 - ((end_x - start_x) * (end_y - start_y) / (H * W))

        batch[:, :, start_x:end_x, start_y:end_y] = batch_roll[:, :, start_x:end_x, start_y:end_y]
        target = target * (1 - ratio) + target_roll * ratio
        return batch, target


def get_dataset(cfg):
    if cfg.dataset.name == 'cifar10':
        dataset_class = CIFAR10
    else:
        dataset_class = CIFAR100
    aug = cfg.dataset.augmentation
    interpolation = aug.train_interpolation
    train_transform = TrainTransform(cfg.dataset.size, aug.train_resize_mode, aug.random_crop_pad,
                                     tuple(aug.random_crop_scale), tuple(aug.random_crop_ratio), aug.hflip,
                                     aug.auto_aug, aug.remode, interpolation, tuple(aug.mean), tuple(aug.std))

    val_transform = ValTransform(cfg.dataset.size, aug.test_resize_mode, aug.crop_pct, interpolation, tuple(aug.mean),
                                 tuple(aug.std))

    train_dataset = dataset_class(root=cfg.dataset.root, train=True, download=False, transform=train_transform)
    val_dataset = dataset_class(root=cfg.dataset.root, train=False, download=False, transform=val_transform)

    return (train_dataset, val_dataset)


def get_cifar_dataloader(cfg):
    aug = cfg.dataset.augmentation
    train_dataset, val_dataset = get_dataset(cfg)

    if cfg.distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = RandomSampler(train_dataset)
        val_sampler = SequentialSampler(val_dataset)

    mix_collate = []
    if aug.mixup:
        mix_collate.append(MixUP(alpha=aug.mixup, nclass=cfg.dataset.num_classes))
    if aug.cutmix:
        mix_collate.append(CutMix(alpha=aug.cutmix, nclass=cfg.dataset.num_classes))

    if mix_collate:
        mix_collate = RandomChoice(mix_collate)
        collate_fn = lambda batch: mix_collate(*default_collate(batch))
    else:
        collate_fn = None

    generator = torch.Generator()
    generator.manual_seed(0)

    # 3. create dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=False, sampler=train_sampler,
                                  num_workers=cfg.train.num_workers, collate_fn=collate_fn, pin_memory=True,
                                  worker_init_fn=partial(_worker_init, worker_seeding='all'),
                                  persistent_workers=True,
                                  generator=generator)

    val_dataloader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False, sampler=val_sampler,
                                num_workers=cfg.train.num_workers, collate_fn=None, pin_memory=False,
                                worker_init_fn=partial(_worker_init, worker_seeding='all'),
                                persistent_workers=True,
                                generator=generator)

    return train_dataloader, val_dataloader
