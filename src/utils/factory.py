import torch
from omegaconf import OmegaConf
from timm.loss import BinaryCrossEntropy, SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from timm.utils import NativeScaler
from torch import nn
from torch.nn import BCEWithLogitsLoss

from src.models import SubResNet


def create_model_cls(model_name, in_channels, num_classes, **kwargs):
    kwargs['in_channels'] = in_channels
    kwargs['num_classes'] = num_classes

    if 'resnet' in model_name or 'resnext' in model_name:
        model_cls = SubResNet
    else:
        raise NotImplementedError(f'{model_name} is not implemented')
    return model_cls(model_name, **kwargs)


class ObjectFactory:
    def __init__(self, cfg):
        self.cfg = cfg
        self.train = cfg.train
        self.optim = cfg.train.optimizer
        self.scheduler = cfg.train.lr_scheduler
        self.dataset = cfg.dataset
        self.model = cfg.model
        self.device = torch.device(f'cuda:{cfg.local_rank}') if torch.cuda.is_available() else torch.device('cpu')

    def create_model(self):
        model = create_model_cls(
            **self.model,
            in_channels=self.dataset.in_channels,
            num_classes=self.dataset.num_classes,
            stem_type=self.dataset.name,
        )
        model.to(self.device)

        if self.train.channels_last:
            model = model.to(memory_format=torch.channels_last)

        if self.cfg.distributed and self.train.sync_bn:
            self.train.dist_bn = ''
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        if self.model.scriptable:
            assert not self.train.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
            model = torch.jit.script(model)
        return model

    def create_optimizer_and_scheduler(self, cfg, model, iter_per_epoch):
        self.cfg.train.iter_per_epoch = iter_per_epoch
        self.train.iter_per_epoch = iter_per_epoch

        if 'cifar' in cfg.dataset.name:
            filter_bias_and_bn = True
        else:
            filter_bias_and_bn = True

        optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=self.optim),
                                        filter_bias_and_bn=filter_bias_and_bn)
        print(optimizer)

        updates_per_epoch = \
            (iter_per_epoch + self.optim.grad_accumulation - 1) // self.optim.grad_accumulation

        OmegaConf.set_struct(self.scheduler, False)
        self.scheduler.epochs = self.train.epochs
        self.scheduler.sched_on_updates = True
        scheduler, _ = create_scheduler_v2(
            optimizer,
            **scheduler_kwargs(self.scheduler),
            updates_per_epoch=updates_per_epoch,
        )

        return optimizer, scheduler

    def create_criterion_scaler(self):
        if self.dataset.augmentation.cutmix > 0 or self.dataset.augmentation.cutmix > 0:
            if self.train.bce_loss:
                train_loss_fn = BinaryCrossEntropy(target_threshold=self.train.bce_target_thresh)
            else:
                train_loss_fn = SoftTargetCrossEntropy()

        elif self.dataset.augmentation.smoothing > 0:
            if self.train.bce_loss:
                train_loss_fn = BinaryCrossEntropy(smoothing=self.dataset.augmentation.smoothing,
                                                   target_threshold=self.train.bce_target_thresh)
            else:
                train_loss_fn = LabelSmoothingCrossEntropy(smoothing=self.dataset.augmentation.smoothing)

        else:
            train_loss_fn = nn.CrossEntropyLoss()
        train_loss_fn = train_loss_fn.to(device=self.device)
        validate_loss_fn = nn.CrossEntropyLoss().to(device=self.device)

        if self.train.amp:
            scaler = NativeScaler()
        else:
            scaler = None

        return (train_loss_fn, validate_loss_fn), scaler


class BCEWithLogitsLossWithTypeCasting(BCEWithLogitsLoss):
    def forward(self, y_hat, y):
        y = y.float()
        y = y.reshape(y_hat.shape)
        return super().forward(y_hat, y)
