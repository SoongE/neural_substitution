import torch
from lion_pytorch import Lion
from timm.loss import BinaryCrossEntropy, SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from timm.optim import Lamb
from timm.utils import NativeScaler
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR, LambdaLR, MultiStepLR, \
    OneCycleLR, SequentialLR

from src.models import SubResNet, AddResNet, SubMobileNet, AddMobileNet
from src.models.mobileone import mobileone


def create_model_cls(model_name, in_channels, num_classes, **kwargs):
    kwargs['in_channels'] = in_channels
    kwargs['num_classes'] = num_classes

    if 'resnet' in model_name or 'resnext' in model_name:
        if 'Sub' in model_name:
            model_cls = SubResNet
        else:
            model_cls = AddResNet
    elif 'mobilenet' in model_name:
        if 'Sub' in model_name:
            model_cls = SubMobileNet
        else:
            model_cls = AddMobileNet
    elif 'mobileone' in model_name:
        if 'Sub' in model_name:
            model_cls = mobileone
            kwargs.update({'substitution': True})
        else:
            model_cls = mobileone
            kwargs.update({'substitution': False})
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

    def create_optimizer_and_scheduler(self, model, iter_per_epoch):
        self.cfg.train.iter_per_epoch = iter_per_epoch
        self.train.iter_per_epoch = iter_per_epoch

        optim = self.optim.optim
        sched = self.scheduler.sched

        parameter = model.parameters()
        total_iter = self.train.epochs * self.train.iter_per_epoch
        warmup_iter = self.scheduler.warmup_epochs * self.train.iter_per_epoch
        lr = self.optim.lr
        weight_decay = self.optim.weight_decay

        if optim == 'sgd':
            optimizer = SGD(parameter, lr, self.optim.momentum, weight_decay=weight_decay, nesterov=self.optim.nesterov)
        elif optim == 'adamw':
            optimizer = AdamW(parameter, lr, weight_decay=weight_decay, betas=self.optim.betas, eps=self.optim.eps)
        elif optim == 'lion':
            optimizer = Lion(parameter, lr, weight_decay=self.optim.weight_decay)
        elif optim == 'lamb':
            optimizer = Lamb(parameter, lr, weight_decay=weight_decay, betas=self.optim.betas, eps=self.optim.eps)
        else:
            NotImplementedError(f"{optim} is not supported yet")

        if sched == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, total_iter - warmup_iter, self.scheduler.min_lr)
        elif sched == 'multistep':
            scheduler = MultiStepLR(optimizer, [epoch * iter_per_epoch for epoch in self.scheduler.milestones],
                                    self.scheduler.gamma)
        elif sched == 'step':
            scheduler = StepLR(optimizer, total_iter - warmup_iter, gamma=self.scheduler.decay_rate)
        elif sched == 'explr':
            scheduler = ExponentialLR(optimizer, gamma=self.scheduler.decay_rate)
        elif sched == 'onecyclelr':
            scheduler = OneCycleLR(optimizer, lr, total_iter)
        else:
            NotImplementedError(f"{sched} is not supported yet")

        if self.scheduler.warmup_epochs and sched != 'onecyclelr':
            if self.scheduler.warmup_scheduler == 'linear':
                lr_lambda = lambda e: (e * (
                        lr - self.scheduler.warmup_lr) / warmup_iter + self.scheduler.warmup_lr) / lr
                warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
            else:
                NotImplementedError(f"{self.scheduler.warmup_scheduler} is not supported yet")

            scheduler = SequentialLR(optimizer, [warmup_scheduler, scheduler], [warmup_iter])

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
