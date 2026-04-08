import os
from torchvision import datasets
from torch.utils.data import DataLoader
from timm.data import create_transform

def get_dataloaders(config):
    """
    基於 timm 的數據加載器實現。
    使用 torch DataLoader 確保 drop_last / persistent_workers 相容性，
    搭配 timm create_transform 提供完整的訓練增強管道。
    """
    mean = tuple(config.get('mean', [0.485, 0.456, 0.406]))
    std  = tuple(config.get('std',  [0.229, 0.224, 0.225]))

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
        mean=mean,
        std=std,
    )

    val_transform = create_transform(
        input_size=config['input_size'],
        is_training=False,
        use_prefetcher=False,
        interpolation=config.get('interpolation', 'bicubic'),
        mean=mean,
        std=std,
    )

    train_dataset = datasets.ImageFolder(
        root=os.path.join(config['data_dir'], 'train'),
        transform=train_transform,
    )
    val_dataset = datasets.ImageFolder(
        root=os.path.join(config['data_dir'], 'val'),
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True,
        persistent_workers=config['num_workers'] > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=False,
        persistent_workers=config['num_workers'] > 0,
    )

    return train_loader, val_loader, len(train_dataset.classes)
