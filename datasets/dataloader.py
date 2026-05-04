import os
import re
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from timm.data import create_transform
from timm.data.loader import PrefetchLoader, fast_collate

try:
    import nvidia.dali.fn as dali_fn
    import nvidia.dali.types as dali_types
    from nvidia.dali.pipeline import pipeline_def
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
    from nvidia.dali.auto_aug.rand_augment import rand_augment as dali_rand_augment
    _DALI_AVAILABLE = True
except Exception:
    dali_fn = None
    dali_types = None
    pipeline_def = None
    DALIGenericIterator = None
    LastBatchPolicy = None
    dali_rand_augment = None
    _DALI_AVAILABLE = False

if pipeline_def is None:
    def pipeline_def(*_args, **_kwargs):
        def _decorator(func):
            return func
        return _decorator


class _DALITupleLoader:
    """Wrap DALIGenericIterator to return (inputs, targets) tuples expected by Trainer."""

    def __init__(self, iterator):
        self.iterator = iterator

    def __iter__(self):
        for batch in self.iterator:
            data = batch[0]
            inputs = data['inputs']
            targets = data['targets'].squeeze(-1)
            yield inputs, targets

    def __len__(self):
        return len(self.iterator)


def _parse_rand_augment(auto_augment):
    """Parse timm-style rand augment string, e.g. rand-m9-mstd0.5-inc1."""
    if not auto_augment:
        return None

    match = re.fullmatch(r'rand-m(\d+)-mstd([0-9]*\.?[0-9]+)-inc1', auto_augment)
    if match is None:
        raise ValueError(
            f"DALI path currently supports only auto_augment='rand-m9-mstd0.5-inc1' style, got: {auto_augment}"
        )

    magnitude = int(match.group(1))
    magnitude_std = float(match.group(2))
    return {
        'n': 2,
        'magnitude': magnitude,
        'magnitude_std': magnitude_std,
        'increasing': True,
    }


def _apply_dali_rand_augment(images, auto_aug_cfg):
    """Apply DALI RandAugment and keep compatibility across DALI versions."""
    if auto_aug_cfg is None:
        return images
    if dali_rand_augment is None:
        raise RuntimeError('DALI RandAugment is unavailable in the installed DALI build.')

    # Different DALI versions expose different rand_augment signatures.
    # Try from most specific (timm-aligned) to most compatible.
    call_candidates = [
        {
            'n': auto_aug_cfg['n'],
            'm': auto_aug_cfg['magnitude'],
            'mstd': auto_aug_cfg['magnitude_std'],
            'monotonic_mag': auto_aug_cfg['increasing'],
            'fill_value': 128,
        },
        {
            'n': auto_aug_cfg['n'],
            'm': auto_aug_cfg['magnitude'],
            'mstd': auto_aug_cfg['magnitude_std'],
            'increasing': auto_aug_cfg['increasing'],
            'fill_value': 128,
        },
        {
            'n': auto_aug_cfg['n'],
            'm': auto_aug_cfg['magnitude'],
            'mstd': auto_aug_cfg['magnitude_std'],
            'fill_value': 128,
        },
        {
            'n': auto_aug_cfg['n'],
            'm': auto_aug_cfg['magnitude'],
            'magnitude_std': auto_aug_cfg['magnitude_std'],
            'fill_value': 128,
        },
        {
            'n': auto_aug_cfg['n'],
            'm': auto_aug_cfg['magnitude'],
            'fill_value': 128,
        },
        {
            'n': auto_aug_cfg['n'],
            'm': auto_aug_cfg['magnitude'],
        },
    ]

    last_error = None
    for kwargs in call_candidates:
        try:
            return dali_rand_augment(images, **kwargs)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        'The installed DALI rand_augment signature is incompatible with current settings. '
        f'Last error: {last_error}'
    )


@pipeline_def(enable_conditionals=True)
def _dali_train_pipeline(
    data_root,
    decoder_device,
    crop_size,
    resize_scale,
    resize_ratio,
    mean,
    std,
    hflip,
    vflip,
    color_jitter,
    auto_aug_cfg,
    dali_output_dtype,
):
    images, labels = dali_fn.readers.file(
        file_root=data_root,
        random_shuffle=True,
        name='Reader',
        pad_last_batch=False,
    )
    images = dali_fn.decoders.image(images, device=decoder_device, output_type=dali_types.RGB)
    if decoder_device == 'cpu':
        images = images.gpu()
    images = dali_fn.random_resized_crop(
        images,
        device='gpu',
        size=crop_size,
        random_area=resize_scale,
        random_aspect_ratio=resize_ratio,
        interp_type=dali_types.INTERP_CUBIC,
    )

    images = _apply_dali_rand_augment(images, auto_aug_cfg)

    if color_jitter and color_jitter > 0:
        cj = float(color_jitter)
        images = dali_fn.color_twist(
            images,
            brightness=dali_fn.random.uniform(range=[1.0 - cj, 1.0 + cj]),
            contrast=dali_fn.random.uniform(range=[1.0 - cj, 1.0 + cj]),
            saturation=dali_fn.random.uniform(range=[1.0 - cj, 1.0 + cj]),
            hue=dali_fn.random.uniform(range=[-0.5 * cj, 0.5 * cj]),
        )

    if hflip and hflip > 0:
        images = dali_fn.flip(images, horizontal=dali_fn.random.coin_flip(probability=float(hflip)))
    if vflip and vflip > 0:
        images = dali_fn.flip(images, vertical=dali_fn.random.coin_flip(probability=float(vflip)))

    images = dali_fn.crop_mirror_normalize(
        images,
        device='gpu',
        dtype=dali_output_dtype,
        output_layout='CHW',
        mean=[m * 255.0 for m in mean],
        std=[s * 255.0 for s in std],
    )

    labels = dali_fn.cast(labels, dtype=dali_types.INT64)
    return images, labels


@pipeline_def(enable_conditionals=True)
def _dali_val_pipeline(
    data_root,
    decoder_device,
    crop_size,
    resize_shorter,
    mean,
    std,
    dali_output_dtype,
):
    images, labels = dali_fn.readers.file(
        file_root=data_root,
        random_shuffle=False,
        name='Reader',
        pad_last_batch=False,
    )
    images = dali_fn.decoders.image(images, device=decoder_device, output_type=dali_types.RGB)
    if decoder_device == 'cpu':
        images = images.gpu()
    images = dali_fn.resize(
        images,
        device='gpu',
        resize_shorter=resize_shorter,
        interp_type=dali_types.INTERP_CUBIC,
    )
    images = dali_fn.crop_mirror_normalize(
        images,
        device='gpu',
        crop=crop_size,
        crop_pos_x=0.5,
        crop_pos_y=0.5,
        dtype=dali_output_dtype,
        output_layout='CHW',
        mean=[m * 255.0 for m in mean],
        std=[s * 255.0 for s in std],
    )
    labels = dali_fn.cast(labels, dtype=dali_types.INT64)
    return images, labels


def _build_dali_loaders(config):
    if not _DALI_AVAILABLE:
        raise ImportError(
            'NVIDIA DALI is not installed. Install a Python 3.8 compatible build, '
            'for example nvidia-dali-cuda120==1.38.0 (adjust for your CUDA version).'
        )

    if not torch.cuda.is_available():
        raise RuntimeError('DALI acceleration requires CUDA GPU, but CUDA is unavailable.')

    mean = tuple(config.get('mean', [0.485, 0.456, 0.406]))
    std = tuple(config.get('std', [0.229, 0.224, 0.225]))
    in_ch, in_h, in_w = config['input_size']
    if in_ch != 3:
        raise ValueError(f'DALI pipeline currently expects 3-channel RGB input, got input_size={config["input_size"]}')

    auto_aug_cfg = _parse_rand_augment(config.get('auto_augment', 'rand-m9-mstd0.5-inc1'))
    dali_output_dtype = dali_types.FLOAT16 if config.get('use_amp', True) else dali_types.FLOAT

    resize_shorter = int(config.get('val_resize_size', round(in_h / 0.875)))
    device_id = int(config.get('dali_device_id', 0))
    num_threads = int(config.get('dali_num_threads', max(2, min(8, os.cpu_count() or 4))))
    seed = int(config.get('dali_seed', 42))
    decoder_device = str(config.get('dali_decoder_device', 'cpu')).lower()
    if decoder_device not in ('cpu', 'mixed'):
        raise ValueError(f"dali_decoder_device must be 'cpu' or 'mixed', got: {decoder_device}")

    train_pipe = _dali_train_pipeline(
        batch_size=config['batch_size'],
        num_threads=num_threads,
        device_id=device_id,
        seed=seed,
        data_root=os.path.join(config['data_dir'], 'train'),
        decoder_device=decoder_device,
        crop_size=(in_h, in_w),
        resize_scale=tuple(config.get('scale', (0.08, 1.0))),
        resize_ratio=tuple(config.get('ratio', (3.0 / 4.0, 4.0 / 3.0))),
        mean=mean,
        std=std,
        hflip=float(config.get('hflip', 0.5)),
        vflip=float(config.get('vflip', 0.0)),
        color_jitter=float(config.get('color_jitter', 0.4)),
        auto_aug_cfg=auto_aug_cfg,
        dali_output_dtype=dali_output_dtype,
    )
    train_pipe.build()

    val_batch_size = int(config.get('val_batch_size', config['batch_size']))
    val_pipe = _dali_val_pipeline(
        batch_size=val_batch_size,
        num_threads=num_threads,
        device_id=device_id,
        seed=seed,
        data_root=os.path.join(config['data_dir'], 'val'),
        decoder_device=decoder_device,
        crop_size=(in_h, in_w),
        resize_shorter=resize_shorter,
        mean=mean,
        std=std,
        dali_output_dtype=dali_output_dtype,
    )
    val_pipe.build()

    # DALI iterator keeps state across epochs (auto_reset=True), equivalent to persistent workers behavior.
    # drop_last is controlled via LastBatchPolicy.DROP/PARTIAL to mirror train/val settings.
    train_iter = DALIGenericIterator(
        [train_pipe],
        output_map=['inputs', 'targets'],
        reader_name='Reader',
        auto_reset=True,
        last_batch_policy=LastBatchPolicy.DROP,
    )
    val_iter = DALIGenericIterator(
        [val_pipe],
        output_map=['inputs', 'targets'],
        reader_name='Reader',
        auto_reset=True,
        last_batch_policy=LastBatchPolicy.PARTIAL,
    )

    class_names = sorted(
        d for d in os.listdir(os.path.join(config['data_dir'], 'train'))
        if os.path.isdir(os.path.join(config['data_dir'], 'train', d))
    )
    return _DALITupleLoader(train_iter), _DALITupleLoader(val_iter), len(class_names)

def get_dataloaders(config):
    if config.get('use_dali', False):
        return _build_dali_loaders(config)

    return _build_timm_prefetch_loaders(config)


def _build_timm_prefetch_loaders(config):
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
