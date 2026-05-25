# FastViT Training Framework

A robust training framework for FastViT models using `timm` and `PyTorch`. This repository provides a complete pipeline for training, evaluating, and inferring with FastViT models, including environment setup via Docker.

## Project Structure

```text
fastvit_trainer/
├── configs/            # Configuration files (YAML) for training and tuning settings
│   └── tune_config.yaml  # Optuna search space, pruner, and tuning output settings
├── datasets/           # Data loading and augmentation logic (timm-based)
├── engine/             # Core training and evaluation loops
├── models/             # Model builder and architecture wrappers
├── utils/              # Utility functions (logging, checkpoints, optimizers)
├── train.py            # Main training entry point
├── tune.py             # Optuna hyperparameter tuning entry point
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
- **MLflow Tracking**: Optional experiment tracking for params, metrics, logs, and model artifacts.
- **Optuna Hyperparameter Tuning**: Automated hyperparameter search integrated with `run_training`, with configurable sampler strategy (e.g., TPE, Random) and optimization history visualization support.
- **ONNX Export**: Export trained checkpoints to ONNX with dynamic batch axes, metadata, optional BatchNorm folding, and optional graph simplification.
- **Dockerized**: Ready-to-use environment with all dependencies pre-installed.
- **NVIDIA DALI (Optional)**: GPU-side decode/resize/crop pipeline for higher throughput.

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

# Build the image with nvidia-dali branch
docker build -t fastvit_trainer:nvidia-dali .

# Run training and shut down TensorBoard when training exits
./run_training_stack.sh

# Windows PowerShell
.\run_training_stack.ps1
```

The wrapper starts Docker Compose with `--abort-on-container-exit --exit-code-from fastvit_trainer` and always runs `docker compose down` when the training container finishes.

Docker Compose in this repo also reads variables from `.env` automatically.

Example `.env`:

```dotenv
APP_TAG=mlflow
IMAGE_NAME=fastvit_trainer
APP_CMD="python3 train.py --config configs/base_config_fastvit_sa12.apple_in1k.yaml"
```

What each variable does:

- `IMAGE_NAME`: Docker image name used by the `fastvit_trainer` service.
- `APP_TAG`: Docker image tag paired with `IMAGE_NAME`.
- `APP_CMD`: Training command passed to the container.

Common workflow:

```bash
# 1. Update .env
# 2. Start the training stack
./run_training_stack.sh

# PowerShell
.\run_training_stack.ps1
```

For example, to switch configs without editing `docker-compose.yml`, change `APP_CMD` in `.env`:

```dotenv
APP_CMD="python3 train.py --config configs/base_config.yaml --run-name experiment_a"
```

You can verify the resolved values with:

```bash
docker compose config
```

#### Local Installation
```bash
pip install -r requirements.txt
```

#### Enable NVIDIA DALI (Python 3.8)
Install a DALI wheel that matches your CUDA runtime (DALI 1.38 supports Python 3.8):
```bash
pip install nvidia-dali-cuda120==1.38.0
```
If your CUDA version is not 12.x, replace `cuda120` with the matching wheel variant.

### 3. Training
Modify `configs/base_config.yaml` to suit your needs, then run:
```bash
python train.py --config configs/base_config.yaml
```

### 4. Inference
```bash
python infer.py --config configs/base_config.yaml --checkpoint output/model_best.pth.tar --image test.jpg
```

### 5. Export to ONNX
```bash
python export_onnx.py \
    --config configs/base_config.yaml \
    --checkpoint output/model_best.pth.tar \
    --output output/fastvit.onnx \
    --fold-bn \
    --simplify
```

```bash
# one line
python export_onnx.py --config configs/base_config_fastvit_sa12.apple_in1k.yaml --checkpoint output/20260409_073421_fastvit_sa12-apple_in1k/checkpoints/model_best.pth.tar --output output/20260409_073421_fastvit_sa12-apple_in1k/fastvit-nc9-RealWaste-sa12_names_fbn_sim.onnx --fold-bn --simplify --opset-version 17
```

The ONNX export script supports:

- dynamic batch axes for input and output
- configurable opset via `--opset-version`
- automatic `num_classes` inference from the checkpoint head
- metadata injection for `model_name`, `num_classes`, `opset_version`, and `class_names`
- optional BatchNorm export prep via `--fold-bn`, which first runs model-specific `reparameterize()` hooks when available and then folds supported adjacent `Conv/Linear + BatchNorm` pairs in eval mode to reduce `BatchNormalization` ops in the exported graph
- optional graph simplification via `--simplify`, which runs `onnx-simplifier` after export to remove redundant graph structure and reduce inference overhead

For FastViT attention variants, a small number of `BatchNormalization` nodes can still remain after `--fold-bn`. Those layers are typically attention pre-norm blocks rather than foldable `Conv/Linear + BatchNorm` pairs.

### 6. Hyperparameter Tuning (Optuna)

Run a minimal tuning session:

```bash
python tune.py --base-config configs/base_config.yaml --tune-config configs/tune_config.yaml --n-trials 10 --epochs 5
```

Use a different sampler (example: RandomSampler) and visualize optimization history:

```python
import optuna
from optuna.visualization import plot_optimization_history

study = optuna.create_study(
    study_name="fastvit_tune",
    storage="sqlite:///output/tune/optuna.db",
    load_if_exists=True,
    sampler=optuna.samplers.RandomSampler(),
    direction="maximize",
)

fig = plot_optimization_history(study)
fig.show()
```

## Configuration

Key parameters in `configs/base_config.yaml`:
- `model_name`: Choose from `fastvit_t8`, `fastvit_ma36`, etc.
- `lr`: Learning rate (default: 0.001).
- `batch_size`: Number of images per GPU batch.
- `use_amp`: Enable/disable mixed precision training.
- `use_dali`: Enable NVIDIA DALI dataloader path.
- `val_resize_size`: Validation resize short side before center crop (default: 256).

### MLflow

Enable MLflow in the config to track training metadata:

```yaml
mlflow:
    enabled: true
    experiment_name: fastvit-trainer
    tracking_uri: http://127.0.0.1:5000
    log_model: true
    log_checkpoints: false
    tags:
        project: fastvit
        dataset: RealWaste
```

When enabled, the trainer logs:

- flattened config params
- per-epoch train and validation metrics
- the copied run config and `train.log`
- the final PyTorch model artifact when `log_model: true`
- checkpoint artifacts when `log_checkpoints: true`

Start a local MLflow server if needed:

```bash
mlflow server --host 0.0.0.0 --port 5000
```

## Acknowledgements
- [timm](https://github.com/huggingface/pytorch-image-models)
- [FastViT Paper](https://arxiv.org/abs/2303.14189)
