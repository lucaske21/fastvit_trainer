import argparse
import json
import os
import torch
from models.builder import create_fastvit_model
from utils.utils import load_config


def safe_torch_load(path, device):
    """Load checkpoint with forward-compatible torch.load options."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        # Older torch versions do not support weights_only
        return torch.load(path, map_location=device)


def infer_num_classes(state_dict, default_num_classes):
    """Infer classifier output dimension from known head keys."""
    candidate_keys = [
        'head.fc.weight',
        'head.weight',
        'classifier.weight',
        'fc.weight',
    ]
    for key in candidate_keys:
        weight = state_dict.get(key)
        if weight is not None and hasattr(weight, 'shape') and len(weight.shape) >= 1:
            return int(weight.shape[0])
    return int(default_num_classes)


def resolve_class_names(args, config):
    """Resolve class names from explicit arg or dataset directory."""
    if args.class_names:
        if os.path.isfile(args.class_names):
            if args.class_names.lower().endswith('.json'):
                with open(args.class_names, 'r', encoding='utf-8') as f:
                    names = json.load(f)
                if not isinstance(names, list):
                    raise ValueError('class names JSON must be a list of strings')
                return [str(x) for x in names]
            with open(args.class_names, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        return [x.strip() for x in args.class_names.split(',') if x.strip()]

    data_dir = config.get('data_dir')
    if data_dir:
        train_dir = os.path.join(data_dir, 'train')
        if os.path.isdir(train_dir):
            names = [d for d in sorted(os.listdir(train_dir)) if os.path.isdir(os.path.join(train_dir, d))]
            if names:
                return names
    return None


def build_extra_files(class_names, num_classes, model_name):
    """Build TorchScript extra files metadata."""
    metadata = {
        'model_name': str(model_name),
        'num_classes': int(num_classes),
        'class_names': class_names if class_names is not None else [],
    }
    return {'metadata.json': json.dumps(metadata, ensure_ascii=False)}


def main():
    parser = argparse.ArgumentParser(description='Export FastViT model to TorchScript')
    parser.add_argument('--config', default='configs/base_config.yaml', type=str, help='Path to config file')
    parser.add_argument('--checkpoint', required=True, type=str, help='Path to model checkpoint (.pth.tar)')
    parser.add_argument('--output', default='output/fastvit.ts', type=str, help='Output TorchScript file path')
    parser.add_argument('--num-classes', default=None, type=int, help='Override number of classes for model head')
    parser.add_argument('--class-names', default=None, type=str,
                        help='Class names source: comma-separated string or path to .json/.txt')
    parser.add_argument('--method', default='trace', choices=['trace', 'script'],
                        help='Export method: trace (default) or script')
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device('cpu')

    checkpoint = safe_torch_load(args.checkpoint, device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    num_classes = args.num_classes
    if num_classes is None:
        num_classes = infer_num_classes(state_dict, config.get('num_classes', 1000))
    class_names = resolve_class_names(args, config)

    export_config = dict(config)
    export_config['pretrained'] = False

    model = create_fastvit_model(export_config, num_classes).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    c, h, w = config.get('input_size', [3, 224, 224])
    dummy_input = torch.randn(1, c, h, w, device=device)

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)

    with torch.no_grad():
        if args.method == 'script':
            ts_model = torch.jit.script(model)
        else:
            ts_model = torch.jit.trace(model, dummy_input)

    extra_files = build_extra_files(class_names, num_classes, export_config.get('model_name', ''))
    ts_model.save(args.output, _extra_files=extra_files)

    print(f"TorchScript model exported to: {args.output}")


if __name__ == '__main__':
    main()
