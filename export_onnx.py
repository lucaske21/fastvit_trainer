import argparse
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

def main():
    parser = argparse.ArgumentParser(description='Export FastViT model to ONNX (opset 11)')
    parser.add_argument('--config', default='configs/base_config.yaml', type=str, help='Path to config file')
    parser.add_argument('--checkpoint', required=True, type=str, help='Path to model checkpoint (.pth.tar)')
    parser.add_argument('--output', default='output/fastvit.onnx', type=str, help='Output ONNX file path')
    parser.add_argument('--num-classes', default=None, type=int, help='Override number of classes for model head')
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device('cpu')

    checkpoint = safe_torch_load(args.checkpoint, device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    num_classes = args.num_classes
    if num_classes is None:
        num_classes = infer_num_classes(state_dict, config.get('num_classes', 1000))

    export_config = dict(config)
    # Export should not trigger downloading pretrained weights.
    export_config['pretrained'] = False

    model = create_fastvit_model(export_config, num_classes).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    c, h, w = config.get('input_size', [3, 224, 224])
    dummy_input = torch.randn(1, c, h, w, device=device)

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        },
        do_constant_folding=True,
    )
    print(f"ONNX model exported to: {args.output}")

if __name__ == '__main__':
    main()
