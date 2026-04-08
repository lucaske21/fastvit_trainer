import argparse
import os
import torch
from models.builder import create_fastvit_model
from utils.utils import load_config

def main():
    parser = argparse.ArgumentParser(description='Export FastViT model to ONNX (opset 11)')
    parser.add_argument('--config', default='configs/base_config.yaml', type=str, help='Path to config file')
    parser.add_argument('--checkpoint', required=True, type=str, help='Path to model checkpoint (.pth.tar)')
    parser.add_argument('--output', default='output/fastvit.onnx', type=str, help='Output ONNX file path')
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device('cpu')

    model = create_fastvit_model(config, config['num_classes']).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
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
