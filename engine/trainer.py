import torch
import torch.nn as nn
from tqdm import tqdm
from timm.utils import accuracy, AverageMeter
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

class Trainer:
    """
    模型訓練與驗證引擎。
    """
    def __init__(self, model, optimizer, scheduler, criterion, device, config):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.config = config
        self.scaler = torch.cuda.amp.GradScaler() if config.get('use_amp', True) else None

    def train_one_epoch(self, train_loader, epoch):
        self.model.train()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        pbar = tqdm(train_loader, desc=f'Epoch {epoch} Training')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

            # 記錄指標
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))

            pbar.set_postfix(loss=losses.avg, top1=top1.avg, top5=top5.avg)

        if self.scheduler is not None:
            self.scheduler.step(epoch)

        return losses.avg, top1.avg, top5.avg

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        pbar = tqdm(val_loader, desc='Validating')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))

            pbar.set_postfix(loss=losses.avg, top1=top1.avg, top5=top5.avg)

        return losses.avg, top1.avg, top5.avg
