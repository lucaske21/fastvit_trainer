import os
import torch
import yaml
import logging
from torch.utils.tensorboard import SummaryWriter

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('FastViT-Trainer')

def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth.tar'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(state, os.path.join(checkpoint_dir, filename))
    if is_best:
        torch.save(state, os.path.join(checkpoint_dir, 'model_best.pth.tar'))

def get_optimizer_scheduler(model, config):
    """
    基於 timm 推薦配置。
    """
    from timm.optim import create_optimizer_v2
    from timm.scheduler import create_scheduler_v2

    optimizer = create_optimizer_v2(
        model,
        opt=config.get('opt', 'adamw'),
        lr=config.get('lr', 1e-3),
        weight_decay=config.get('weight_decay', 0.05),
        momentum=config.get('momentum', 0.9)
    )

    scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        sched=config.get('sched', 'cosine'),
        num_epochs=config.get('epochs', 300),
        decay_epochs=config.get('decay_epochs', 30),
        decay_rate=config.get('decay_rate', 0.1),
        warmup_lr=config.get('warmup_lr', 1e-6),
        warmup_epochs=config.get('warmup_epochs', 5),
        cooldown_epochs=config.get('cooldown_epochs', 10),
        min_lr=config.get('min_lr', 1e-5),
        cycle_mul=config.get('cycle_mul', 1.0),
        cycle_decay=config.get('cycle_decay', 0.1),
        cycle_limit=config.get('cycle_limit', 1),
    )

    return optimizer, scheduler
