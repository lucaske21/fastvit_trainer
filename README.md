# FastViT Training Framework

A robust training framework for FastViT models using `timm` and `PyTorch`. This repository provides a complete pipeline for training, evaluating, and inferring with FastViT models, including environment setup via Docker.

## Project Structure

```text
fastvit_trainer/
├── configs/            # Configuration files (YAML) for training settings
├── datasets/           # Data loading and augmentation logic (timm-based)
├── engine/             # Core training and evaluation loops
├── models/             # Model builder and architecture wrappers
├── utils/              # Utility functions (logging, checkpoints, optimizers)
├── train.py            # Main training entry point
├── infer.py            # Inference script for single images
├── Dockerfile          # Docker environment configuration
├── docker-compose.yml  # Docker Compose for easy deployment
└── requirements.txt    # Python package dependencies
```

## Features

- **Timm Integration**: Full support for all FastViT variants available in the `timm` library.
- **Advanced Augmentation**: Includes RandAugment, Mixup, Cutmix, and Random Erasing via `timm`.
- **Flexible Configuration**: All hyperparameters managed through easy-to-read YAML files.
- **AMP Support**: Automatic Mixed Precision training for faster performance on NVIDIA GPUs.
- **Dockerized**: Ready-to-use environment with all dependencies pre-installed.

## Getting Started

### 1. Prepare Data
Organize your dataset in the standard `ImageFolder` format:
```text
data/
├── train/
│   ├── class1/
│   └── class2/
└── val/
    ├── class1/
    └── class2/
```

### 2. Environment Setup
#### Using Docker (Recommended)
```bash
# Build the image
docker build -t fastvit_trainer .

# Run training via Docker Compose
docker-compose up
```

#### Local Installation
```bash
pip install -r requirements.txt
```

### 3. Training
Modify `configs/base_config.yaml` to suit your needs, then run:
```bash
python train.py --config configs/base_config.yaml
```

### 4. Inference
```bash
python infer.py --config configs/base_config.yaml --checkpoint output/model_best.pth.tar --image test.jpg
```

## Configuration

Key parameters in `configs/base_config.yaml`:
- `model_name`: Choose from `fastvit_t8`, `fastvit_ma36`, etc.
- `lr`: Learning rate (default: 0.001).
- `batch_size`: Number of images per GPU batch.
- `use_amp`: Enable/disable mixed precision training.

## Acknowledgements
- [timm](https://github.com/huggingface/pytorch-image-models)
- [FastViT Paper](https://arxiv.org/abs/2303.14189)
