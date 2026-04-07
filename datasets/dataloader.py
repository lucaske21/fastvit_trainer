import os
from torchvision import datasets
from timm.data import create_loader, create_transform

def get_dataloaders(config):
    """
    基於 timm 的數據加載器實現。
    使用 use_prefetcher=True：transform 管道省略 ToTensor/Normalize，
    由 create_loader 的 PrefetchLoader（fast_collate + GPU normalize）接管，
    同時啟用 persistent_workers 消除 epoch 間 worker 重建開銷。
    """
    mean = tuple(config.get('mean', [0.485, 0.456, 0.406]))
    std  = tuple(config.get('std',  [0.229, 0.224, 0.225]))

    # 訓練集：PIL 空間增強，ToTensor/Normalize/RE 移至 GPU PrefetchLoader
    train_transform = create_transform(
        input_size=config['input_size'],
        is_training=True,
        use_prefetcher=True,
        no_aug=False,
        re_prob=0.0,  # RE 交由 PrefetchLoader 在 GPU 執行
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

    # 驗證集：僅 Resize + CenterCrop，同樣省略 Normalize
    val_transform = create_transform(
        input_size=config['input_size'],
        is_training=False,
        use_prefetcher=True,
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

    # create_loader 內部使用 fast_collate（PIL→uint8 tensor）+ PrefetchLoader（GPU normalize）
    train_loader = create_loader(
        train_dataset,
        input_size=config['input_size'],
        batch_size=config['batch_size'],
        is_training=True,
        use_prefetcher=True,
        mean=mean,
        std=std,
        re_prob=config.get('re_prob', 0.25),
        re_mode=config.get('re_mode', 'pixel'),
        re_count=config.get('re_count', 1),
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    val_loader = create_loader(
        val_dataset,
        input_size=config['input_size'],
        batch_size=config['batch_size'],
        is_training=False,
        use_prefetcher=True,
        mean=mean,
        std=std,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )

    return train_loader, val_loader, len(train_dataset.classes)
