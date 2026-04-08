import os
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from timm.data import create_transform
from timm.data.loader import PrefetchLoader, fast_collate

def get_dataloaders(config):
    """
    基於 timm 的數據加載器實現。
    使用 torch DataLoader 確保 drop_last / persistent_workers 相容性，
    並以 timm PrefetchLoader 將 normalize / random erasing 移至 GPU 端。
    """
    mean = tuple(config.get('mean', [0.485, 0.456, 0.406]))
    std  = tuple(config.get('std',  [0.229, 0.224, 0.225]))
    use_gpu_prefetcher = config.get('use_gpu_prefetcher', True) and torch.cuda.is_available()

    train_transform = create_transform(
        input_size=config['input_size'],
        is_training=True,
        use_prefetcher=use_gpu_prefetcher,
        no_aug=False,
        # 使用 PrefetchLoader 時，Random Erasing 在 GPU 端做，避免 CPU 重複開銷
        re_prob=0.0 if use_gpu_prefetcher else config.get('re_prob', 0.25),
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
        use_prefetcher=use_gpu_prefetcher,
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

    common_loader_kwargs = {
        'num_workers': config['num_workers'],
        'pin_memory': True,
        'persistent_workers': config['num_workers'] > 0,
    }
    if config['num_workers'] > 0:
        common_loader_kwargs['prefetch_factor'] = config.get('prefetch_factor', 2)
    if use_gpu_prefetcher:
        common_loader_kwargs['collate_fn'] = fast_collate

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        drop_last=True,
        **common_loader_kwargs,
    )

    val_batch_size = int(config.get('val_batch_size', config['batch_size']))
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        drop_last=False,
        **common_loader_kwargs,
    )

    if use_gpu_prefetcher:
        prefetch_device = torch.device('cuda')
        train_loader = PrefetchLoader(
            train_loader,
            mean=mean,
            std=std,
            channels=config['input_size'][0],
            device=prefetch_device,
            fp16=config.get('use_amp', True) and torch.cuda.is_available(),
            re_prob=config.get('re_prob', 0.25),
            re_mode=config.get('re_mode', 'pixel'),
            re_count=config.get('re_count', 1),
            re_num_splits=0,
        )
        val_loader = PrefetchLoader(
            val_loader,
            mean=mean,
            std=std,
            channels=config['input_size'][0],
            device=prefetch_device,
            fp16=config.get('use_amp', True) and torch.cuda.is_available(),
            re_prob=0.0,
            re_mode='const',
            re_count=1,
            re_num_splits=0,
        )

    return train_loader, val_loader, len(train_dataset.classes)
