import argparse
import os
import shutil
import torch
import torch.nn as nn
from datetime import datetime
from datasets.dataloader import get_dataloaders
from models.builder import create_fastvit_model
from engine.trainer import Trainer
from utils.utils import (
    load_config,
    setup_logger,
    get_optimizer_scheduler,
    AsyncCheckpointSaver,
    MLflowTracker,
)
from torch.utils.tensorboard import SummaryWriter


def make_run_dir(base_output_dir: str, model_name: str, run_name) -> str:
    """Create a unique run directory under base_output_dir."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_model = model_name.replace('/', '_').replace('.', '-')
    folder = f"{timestamp}_{safe_model}"
    if run_name:
        folder = f"{folder}_{run_name}"
    run_dir = os.path.join(base_output_dir, folder)
    os.makedirs(os.path.join(run_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'tensorboard'), exist_ok=True)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description='FastViT Training Framework')
    parser.add_argument('--config', default='configs/base_config.yaml', type=str,
                        help='Path to config file')
    parser.add_argument('--run-name', default=None, type=str,
                        help='Optional tag appended to the run directory name')
    parser.add_argument('--resume', default=None, type=str,
                        help='Path to a checkpoint to resume training from')
    args = parser.parse_args()

    # 加載配置
    config = load_config(args.config)

    if torch.cuda.is_available() and config.get('device', 'cuda') == 'cuda':
        # 固定尺寸輸入時可加速 cudnn kernel 選型，提升整體吞吐
        torch.backends.cudnn.benchmark = True

    # 建立本次訓練的獨立輸出目錄
    run_dir = make_run_dir(config['output_dir'], config['model_name'], args.run_name)

    # 保存本次使用的配置副本，方便復現
    shutil.copy(args.config, os.path.join(run_dir, 'config.yaml'))

    # 初始化日誌和 Tensorboard
    logger = setup_logger(run_dir)
    writer = SummaryWriter(log_dir=os.path.join(run_dir, 'tensorboard'))
    logger.info(f"Run directory: {run_dir}")
    mlflow_tracker = MLflowTracker(config, run_dir, logger, run_name=args.run_name, resume_path=args.resume)

    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    logger.info(f"Using device: {device}")

    # 獲取數據加載器
    train_loader, val_loader, num_classes = get_dataloaders(config)
    logger.info(f"Datasets loaded: {num_classes} classes found.")

    # 創建模型
    model = create_fastvit_model(config, num_classes).to(device)
    logger.info(f"Model {config['model_name']} created.")

    # 定義損失函數、優化器和調度器
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = get_optimizer_scheduler(model, config)

    # 初始化訓練引擎
    trainer = Trainer(model, optimizer, scheduler, criterion, device, config)

    # 斷點續訓
    best_acc1 = 0.0
    start_epoch = 1
    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"Checkpoint not found: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        best_acc1 = checkpoint.get('best_acc1', 0.0)
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resumed from {args.resume} (epoch {checkpoint['epoch']}, best_acc1={best_acc1:.2f}%)")

    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    checkpoint_saver = AsyncCheckpointSaver(checkpoint_dir)

    validate_interval = max(1, int(config.get('validate_interval', 5)))
    checkpoint_interval = max(1, int(config.get('checkpoint_interval', validate_interval)))

    status = 'FAILED'
    try:
        mlflow_tracker.start(config)
        mlflow_tracker.log_artifact(os.path.join(run_dir, 'config.yaml'), artifact_path='configs')

        for epoch in range(start_epoch, config['epochs'] + 1):
            # 訓練一輪
            train_loss, train_acc1, train_acc5 = trainer.train_one_epoch(train_loader, epoch)

            do_validate = (epoch % validate_interval == 0) or (epoch == config['epochs'])
            if do_validate:
                val_loss, val_acc1, val_acc5 = trainer.validate(val_loader)
            else:
                val_loss, val_acc1, val_acc5 = float('nan'), float('nan'), float('nan')

            # 記錄日誌
            if do_validate:
                logger.info(
                    f"Epoch {epoch}/{config['epochs']}: "
                    f"Train Loss={train_loss:.4f} Acc@1={train_acc1:.2f}% | "
                    f"Val Loss={val_loss:.4f} Acc@1={val_acc1:.2f}%"
                )
            else:
                logger.info(
                    f"Epoch {epoch}/{config['epochs']}: "
                    f"Train Loss={train_loss:.4f} Acc@1={train_acc1:.2f}% | "
                    f"Val skipped (validate_interval={validate_interval})"
                )

            # 寫入 Tensorboard
            writer.add_scalar('Loss/train', train_loss, epoch)
            if do_validate:
                writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc1, epoch)
            if do_validate:
                writer.add_scalar('Accuracy/val', val_acc1, epoch)

            current_lr = optimizer.param_groups[0]['lr'] if optimizer.param_groups else 0.0
            metric_payload = {
                'train_loss': train_loss,
                'train_acc1': train_acc1,
                'train_acc5': train_acc5,
                'lr': current_lr,
            }
            if do_validate:
                metric_payload.update({
                    'val_loss': val_loss,
                    'val_acc1': val_acc1,
                    'val_acc5': val_acc5,
                })
            mlflow_tracker.log_metrics(metric_payload, step=epoch)

            # 保存 Checkpoint
            is_best = do_validate and (val_acc1 > best_acc1)
            if do_validate:
                best_acc1 = max(val_acc1, best_acc1)

            should_save = is_best or (epoch % checkpoint_interval == 0) or (epoch == config['epochs'])
            if should_save:
                checkpoint_state = {
                    'epoch': epoch,
                    'model_name': config['model_name'],
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }
                checkpoint_saver.submit(checkpoint_state, is_best)

        mlflow_tracker.log_model_artifact(model)
        status = 'FINISHED'
    finally:
        writer.close()
        checkpoint_saver.close()
        if mlflow_tracker.log_checkpoints:
            for checkpoint_name in sorted(os.listdir(checkpoint_dir)):
                checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
                if os.path.isfile(checkpoint_path):
                    mlflow_tracker.log_artifact(checkpoint_path, artifact_path='checkpoints')
        mlflow_tracker.log_artifact(os.path.join(run_dir, 'train.log'), artifact_path='logs')
        mlflow_tracker.finish(status=status)

    logger.info(f"Training completed. Best Val Acc@1: {best_acc1:.2f}%")
    logger.info(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == '__main__':
    main()
