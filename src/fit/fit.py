import logging

import torch
from timm.models import model_parameters
from timm.utils import dispatch_clip_grad, distribute_bn, update_summary
from torch.cuda.amp import autocast
from torchmetrics import MeanMetric, Accuracy
from torchmetrics.functional import accuracy


class Fit:
    def __init__(self, cfg, scaler, device, epochs, model, criterion, optimizer, model_ema,
                 scheduler, saver, loader=None):
        self.device = device
        self.local_rank = cfg.local_rank
        self.world_size = cfg.world_size

        self.distributed = cfg.distributed
        self.dist_bn = cfg.train.dist_bn
        self.clip_grad = cfg.train.optimizer.clip_grad
        self.clip_mode = cfg.train.optimizer.clip_mode
        self.grad_accumulation = cfg.train.optimizer.grad_accumulation
        self.double_valid = cfg.train.double_valid
        self.wandb = cfg.wandb
        self.start_epoch, self.num_epochs = epochs
        self.logging_interval = 100
        self.num_classes = cfg.dataset.num_classes
        self.tm = cfg.train.target_metric

        self.model = model
        if isinstance(criterion, tuple):
            self.train_criterion = criterion[0]
            self.val_criterion = criterion[1]
        else:
            self.train_criterion = self.val_criterion = criterion
        self.optimizer = optimizer
        self.model_ema = model_ema
        self.scaler = scaler
        self.scheduler = scheduler
        self.saver = saver
        if loader:
            self.train_loader = loader[0]
            self.val_loader = loader[1]

        self.losses = MeanMetric(compute_on_step=False).to(self.device)
        self.top1 = Accuracy(task='multiclass', compute_on_step=False, num_classes=self.num_classes).to(self.device)
        self.top5 = Accuracy(task='multiclass', compute_on_step=False, num_classes=self.num_classes, top_k=5).to(
            self.device)

    def __call__(self, *args, **kwargs):
        for epoch in range(self.start_epoch, self.num_epochs):
            if self.distributed and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            train_metrics = self.train(epoch)
            if self.distributed and self.dist_bn in ('broadcast', 'reduce'):
                distribute_bn(self.model, self.world_size, self.dist_bn == 'reduce')

            if self.model_ema:
                if self.distributed and self.dist_bn in ('broadcast', 'reduce'):
                    distribute_bn(self.model_ema, self.world_size, self.dist_bn == 'reduce')
                eval_metrics = self.validate(epoch, ema=True)
                if self.double_valid:
                    _ = self.validate(epoch, ema=False)
            else:
                eval_metrics = self.validate(epoch, ema=False)

            # save proper checkpoint with eval metric
            if self._master_node():
                torch.cuda.synchronize()
                best_metric, best_epoch = self.saver.save_checkpoint(epoch, metric=eval_metrics[self.tm])
                eval_metrics.update({'Best_Top1': best_metric})
                update_summary(epoch, train_metrics, eval_metrics, 'summary.csv', log_wandb=self.wandb)

            # if epoch + 1 % 10 == 0:
            #     for name, module in self.model.named_modules():
            #         if isinstance(module, BasicBlockSub):
            #             module.conv1.shuffle = (epoch + 1) * 0.1
            #             module.conv2.shuffle = (epoch + 1) * 0.1

        if self._master_node():
            logging.info(f'*** Best Top1: {best_metric} {best_epoch} ***')

    def iterate(self, model, data, criterion):
        x, y = map(lambda x: x.to(self.device), data)
        x = x.to(memory_format=torch.channels_last)

        with autocast(enabled=True if self.scaler else False):
            prob = model(x)
            loss = criterion(prob, y)

        return loss, prob, y

    def train(self, epoch):
        self._reset_metric()
        total = len(self.train_loader)
        second_order = getattr(self.optimizer, 'is_second_order', False)

        self.model.train()
        self.optimizer.zero_grad()
        for i, data in enumerate(self.train_loader):
            loss, prob, target = self.iterate(self.model, data, self.train_criterion)
            loss /= self.grad_accumulation
            self._backward(loss, (i + 1) % self.grad_accumulation == 0, second_order)

            self.losses.update(loss)
            computed_losses = self.losses.compute()
            if self._master_node() and i % self.logging_interval == 0:
                logging.info(f'Train {epoch:>3}: [{i:>4}/{total}]  loss:{computed_losses:.6f}')

            torch.cuda.synchronize()
            self.scheduler.step()

        if hasattr(self.optimizer, 'sync_lookahead'):
            self.optimizer.sync_lookahead()

        return {'loss': self.losses.compute()}

    @torch.no_grad()
    def validate(self, epoch, ema=False):
        model = self.model_ema if ema else self.model
        log_prefix = ' EMA ' if ema else ' Val '
        self._reset_metric()
        total = len(self.val_loader)

        model.eval()
        for i, data in enumerate(self.val_loader):
            loss, prob, target = self.iterate(model, data, self.val_criterion)
            loss /= self.grad_accumulation
            self._update_metric(loss, prob, target)

            metrics = self._metrics()
            if self._master_node() and i % self.logging_interval == 0:
                logging.info(self._print(metrics, epoch, i, total, log_prefix))

        return self._metrics()

    @torch.no_grad()
    def test(self, ema, test_loader):
        self._reset_metric()
        accuracies_list = list()
        total = len(test_loader)

        model = self.model_ema if ema else self.model

        model.eval()
        for i, data in enumerate(test_loader):
            loss, prob, target = self.iterate(model, data, self.val_criterion)
            self._update_metric(loss, prob, target)
            accuracies_list.append(accuracy(prob, target, num_classes=1000, task='multiclass'))

            metrics = self._metrics()
            if self._master_node() and i % self.logging_interval == 0:
                logging.info(self._print(metrics, 0, i, total, 'Test'))

        return accuracies_list, self._metrics()

    def _backward(self, loss, update_grad, second_order):
        if self.scaler:
            self._scaler_backward(loss, update_grad, second_order)
        else:
            self._default_backward(loss, update_grad, second_order)

        if update_grad:
            self.optimizer.zero_grad()
            if self.model_ema:
                self.model_ema.update(self.model)

    def _scaler_backward(self, loss, update_grad, second_order):
        self.scaler(loss, self.optimizer, self.clip_grad, self.clip_mode,
                    model_parameters(self.model, exclude_head='agc' in self.clip_mode), second_order, update_grad)

    def _default_backward(self, loss, update_grad, second_order):
        loss.backward(second_order)
        if update_grad:
            if self.clip_grad:
                dispatch_clip_grad(self.model.parameters(), self.clip_grad, mode=self.clip_mode)
            self.optimizer.step()

    def _update_metric(self, loss, prob, target):
        self.losses.update(loss.item() / prob.size(0))
        self.top1.update(prob, target)
        self.top5.update(prob, target)

    def _reset_metric(self):
        self.losses.reset()
        self.top1.reset()
        self.top5.reset()

    def _metrics(self):
        return {
            'loss': self.losses.compute(),
            'top1': self.top1.compute() * 100,
            'top5': self.top5.compute() * 100,
        }

    def _print(self, metrics, epoch, i, max_iter, mode):
        log = f'{mode} {epoch:>3}: [{i:>4d}/{max_iter}]  '
        for k, v in metrics.items():
            log += f'{k}:{v:.6f} | '
        return log[:-3]

    def _master_node(self):
        return self.local_rank == 0


from torch.nn.parallel import DistributedDataParallel as DDP


class FitForDyREP:
    def __init__(self, cfg, scaler, device, epochs, model, criterion, optimizer, model_ema,
                 scheduler, saver, loader=None):
        self.device = device
        self.local_rank = cfg.local_rank
        self.world_size = cfg.world_size

        self.distributed = cfg.distributed
        self.dist_bn = cfg.train.dist_bn
        self.clip_grad = cfg.train.optimizer.clip_grad
        self.clip_mode = cfg.train.optimizer.clip_mode
        self.grad_accumulation = cfg.train.optimizer.grad_accumulation
        self.double_valid = cfg.train.double_valid
        self.wandb = cfg.wandb
        self.start_epoch, self.num_epochs = epochs
        self.logging_interval = 100
        self.num_classes = cfg.dataset.num_classes
        self.tm = cfg.train.target_metric

        self.model = model[0]
        self.dyrep = model[1]
        self.dyrep.optimizer = optimizer
        if isinstance(criterion, list):
            self.train_criterion = criterion[0]
            self.val_criterion = criterion[1]
        else:
            self.train_criterion = self.val_criterion = criterion
        self.optimizer = optimizer
        self.model_ema = model_ema
        self.scaler = scaler
        self.scheduler = scheduler
        self.saver = saver
        if loader:
            self.train_loader = loader[0]
            self.val_loader = loader[1]

        self.losses = MeanMetric(compute_on_step=False).to(self.device)
        self.top1 = Accuracy(task='multiclass', compute_on_step=False, num_classes=self.num_classes).to(self.device)
        self.top5 = Accuracy(task='multiclass', compute_on_step=False, num_classes=self.num_classes, top_k=5).to(
            self.device)

        self.dyrep_max_adjust_epochs = 500
        self.dyrep_adjust_interval = 5

    def __call__(self, *args, **kwargs):
        for epoch in range(self.start_epoch, self.num_epochs):
            if self.distributed and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            train_metrics = self.train(epoch)
            if self.distributed and self.dist_bn in ('broadcast', 'reduce'):
                distribute_bn(self.model, self.world_size, self.dist_bn == 'reduce')

            if epoch < self.dyrep_max_adjust_epochs:
                if (epoch + 1) % self.dyrep_adjust_interval == 0:
                    self.dyrep.adjust_model()
                    # re-init DDP
                    self.model = DDP(self.model.module,
                                     device_ids=[self.local_rank],
                                     find_unused_parameters=True)

            if self.model_ema:
                if self.distributed and self.dist_bn in ('broadcast', 'reduce'):
                    distribute_bn(self.model_ema, self.world_size, self.dist_bn == 'reduce')
                eval_metrics = self.validate(epoch, ema=True)
                if self.double_valid:
                    _ = self.validate(epoch, ema=False)
            else:
                eval_metrics = self.validate(epoch, ema=False)

            # save proper checkpoint with eval metric
            if self._master_node():
                torch.cuda.synchronize()
                best_metric, best_epoch = self.saver.save_checkpoint(epoch, metric=eval_metrics[self.tm])
                eval_metrics.update({'Best_Top1': best_metric})
                update_summary(epoch, train_metrics, eval_metrics, 'summary.csv', log_wandb=self.wandb)

        if self._master_node():
            logging.info(f'*** Best Top1: {best_metric} {best_epoch} ***')

    def iterate(self, model, data, criterion):
        x, y = map(lambda x: x.to(self.device), data)
        x = x.to(memory_format=torch.channels_last)

        with autocast(enabled=True if self.scaler else False):
            prob = model(x)
            loss = criterion(prob, y)

        return loss, prob, y

    def train(self, epoch):
        self._reset_metric()
        total = len(self.train_loader)
        num_updates = epoch * total

        self.model.train()
        self.optimizer.zero_grad()
        for i, data in enumerate(self.train_loader):
            loss, prob, target = self.iterate(self.model, data, self.train_criterion)
            loss /= self.grad_accumulation

            self.dyrep.record_metrics()
            self._backward(loss, (i + 1) % self.grad_accumulation == 0)

            self.losses.update(loss)
            computed_losses = self.losses.compute()
            if self._master_node() and i % self.logging_interval == 0:
                logging.info(f'Train {epoch:>3}: [{i:>4}/{total}]  loss:{computed_losses:.6f}')

            torch.cuda.synchronize()
            num_updates += 1
            self.scheduler.step()

        if hasattr(self.optimizer, 'sync_lookahead'):
            self.optimizer.sync_lookahead()

        return {'loss': self.losses.compute()}

    @torch.no_grad()
    def validate(self, epoch, ema=False):
        model = self.model_ema if ema else self.model
        log_prefix = ' EMA ' if ema else ' Val '
        self._reset_metric()
        total = len(self.val_loader)

        model.eval()
        for i, data in enumerate(self.val_loader):
            loss, prob, target = self.iterate(model, data, self.val_criterion)
            loss /= self.grad_accumulation
            self._update_metric(loss, prob, target)

            metrics = self._metrics()
            if self._master_node() and i % self.logging_interval == 0:
                logging.info(self._print(metrics, epoch, i, total, log_prefix))

        return self._metrics()

    @torch.no_grad()
    def test(self, ema, test_loader):
        self._reset_metric()
        accuracies_list = list()
        total = len(test_loader)

        model = self.model_ema if ema else self.model

        model.eval()
        for i, data in enumerate(test_loader):
            loss, prob, target = self.iterate(model, data, self.val_criterion)
            self._update_metric(loss, prob, target)
            accuracies_list.append(accuracy(prob, target, num_classes=1000, task='multiclass'))

            metrics = self._metrics()
            if self._master_node() and i % self.logging_interval == 0:
                logging.info(self._print(metrics, 0, i, total, 'Test'))

        return accuracies_list, self._metrics()

    def _backward(self, loss, update_grad):
        if self.scaler:
            self._scaler_backward(loss, update_grad)
        else:
            self._default_backward(loss, update_grad)

        if update_grad:
            self.optimizer.zero_grad()
            if self.model_ema:
                self.model_ema.update(self.model)

    def _scaler_backward(self, loss, update_grad):
        self.scaler(loss, self.optimizer, self.model.parameters(), self.clip_grad, update_grad)

    def _default_backward(self, loss, update_grad):
        loss.backward()
        if update_grad:
            if self.clip_grad:
                dispatch_clip_grad(self.model.parameters(), self.clip_grad, mode=self.clip_mode)
            self.optimizer.step()

    def _update_metric(self, loss, prob, target):
        self.losses.update(loss.item() / prob.size(0))
        self.top1.update(prob, target)
        self.top5.update(prob, target)

    def _reset_metric(self):
        self.losses.reset()
        self.top1.reset()
        self.top5.reset()

    def _metrics(self):
        return {
            'loss': self.losses.compute(),
            'top1': self.top1.compute() * 100,
            'top5': self.top5.compute() * 100,
        }

    def _print(self, metrics, epoch, i, max_iter, mode):
        log = f'{mode} {epoch:>3}: [{i:>4d}/{max_iter}]  '
        for k, v in metrics.items():
            log += f'{k}:{v:.6f} | '
        return log[:-3]

    def _master_node(self):
        return self.local_rank == 0
