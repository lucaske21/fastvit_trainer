import argparse
import os
import torch
import torch.nn as nn
from datasets.dataloader import get_dataloaders
from models.builder import create_fastvit_model
from engine.trainer import Trainer
from utils.utils import load_config, setup_logger, save_checkpoint, get_optimizer_scheduler
from torch.utils.tensorboard import SummaryWriter

def main():
    parser = argparse.ArgumentParser(description='FastViT Training Framework')
    parser.add_argument('--config', default='configs/base_config.yaml', type=str, help='Path to config file')
    args = parser.parse_args()

    # 加載配置
    config = load_config(args.config)
    
    # 初始化日誌和 Tensorboard
    logger = setup_logger(config['output_dir'])
    writer = SummaryWriter(log_dir=os.path.join(config['output_dir'], 'tensorboard'))
    
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

    best_acc1 = 0.0
    for epoch in range(1, config['epochs'] + 1):
        # 訓練一輪
        train_loss, train_acc1, train_acc5 = trainer.train_one_epoch(train_loader, epoch)
        
        # 驗證
        val_loss, val_acc1, val_acc5 = trainer.validate(val_loader)

        # 記錄日誌
        logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc@1: {train_acc1:.2f}%, Val Loss: {val_loss:.4f}, Val Acc@1: {val_acc1:.2f}%")
        
        # 寫入 Tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc1, epoch)
        writer.add_scalar('Accuracy/val', val_acc1, epoch)

        # 保存 Checkpoint
        is_best = val_acc1 > best_acc1
        best_acc1 = max(val_acc1, best_acc1)
        
        save_checkpoint({
            'epoch': epoch,
            'model_name': config['model_name'],
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, is_best, config['output_dir'])

    writer.close()
    logger.info("Training completed.")

if __name__ == '__main__':
    main()
