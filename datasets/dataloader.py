import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timm.data import create_loader, create_dataset, create_transform

def get_dataloaders(config):
    """
    基於 timm 的數據加載器實現。
    """
    # 訓練集數據增強 (基於 timm 推薦配置)
    train_transform = create_transform(
        input_size=config['input_size'],
        is_training=True,
        use_prefetcher=False,
        no_aug=False,
        re_prob=config.get('re_prob', 0.25),
        re_mode=config.get('re_mode', 'pixel'),
        re_count=config.get('re_count', 1),
        scale=config.get('scale', (0.08, 1.0)),
        ratio=config.get('ratio', (3./4., 4./3.)),
        hflip=config.get('hflip', 0.5),
        vflip=config.get('vflip', 0.0),
        color_jitter=config.get('color_jitter', 0.4),
        auto_augment=config.get('auto_augment', 'rand-m9-mstd0.5-inc1'),
        interpolation=config.get('interpolation', 'bicubic'),
        mean=config.get('mean', (0.485, 0.456, 0.406)),
        std=config.get('std', (0.229, 0.224, 0.225)),
    )

    # 驗證集數據增強
    val_transform = create_transform(
        input_size=config['input_size'],
        is_training=False,
        use_prefetcher=False,
        interpolation=config.get('interpolation', 'bicubic'),
        mean=config.get('mean', (0.485, 0.456, 0.406)),
        std=config.get('std', (0.229, 0.224, 0.225)),
    )

    # 加載數據集 (假設使用 ImageFolder 格式)
    train_dataset = datasets.ImageFolder(
        root=os.path.join(config['data_dir'], 'train'),
        transform=train_transform
    )
    
    val_dataset = datasets.ImageFolder(
        root=os.path.join(config['data_dir'], 'val'),
        transform=val_transform
    )

    # 創建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader, len(train_dataset.classes)
