# FastViT 训练框架

这是一个基于 `timm` 和 `PyTorch` 的 FastViT 稳健训练框架。该仓库提供了完整的 FastViT 训练、评估与推理流程，并包含基于 Docker 的环境配置。

## 项目结构

```text
fastvit_trainer/
├── configs/            # 训练配置文件（YAML）
├── datasets/           # 数据加载与数据增强逻辑（基于 timm）
├── engine/             # 核心训练与评估流程
├── models/             # 模型构建与架构封装
├── utils/              # 工具函数（日志、checkpoint、优化器等）
├── train.py            # 主训练入口
├── infer.py            # 单张图片推理脚本
├── Dockerfile          # Docker 环境配置
├── docker-compose.yml  # Docker Compose 部署配置
└── requirements.txt    # Python 依赖列表
```

## 功能特性

- **集成 Timm**：完整支持 `timm` 库中的 FastViT 变体。
- **高级数据增强**：包含 RandAugment、Mixup、Cutmix 和 Random Erasing。
- **灵活配置**：所有超参数通过清晰易读的 YAML 文件管理。
- **AMP 支持**：支持 NVIDIA GPU 上的自动混合精度训练以提升性能。
- **MLflow 跟踪**：可选实验跟踪，记录参数、指标、日志和模型产物。
- **Docker 化**：开箱即用，依赖已预装。
- **NVIDIA DALI（可选）**：支持在 GPU 侧执行 decode/resize/crop 以提升吞吐。

## 快速开始

### 1. 准备数据
请按照标准 `ImageFolder` 格式组织数据集：

```text
data/
├── train/
│   ├── class1/
│   └── class2/
└── val/
    ├── class1/
    └── class2/
```

### 2. 环境配置
#### 使用 Docker（推荐）
```bash
# 构建镜像
docker build -t fastvit_trainer .

# 构建带 nvidia-dali 分支的镜像
docker build -t fastvit_trainer:nvidia-dali .

# 启动训练，训练结束后自动关闭 TensorBoard 和 Compose 服务
./run_training_stack.sh

# Windows PowerShell
.\run_training_stack.ps1
```

这个包装脚本会以 `--abort-on-container-exit --exit-code-from fastvit_trainer` 启动 Docker Compose，并在训练容器结束后自动执行 `docker compose down`。

这个仓库中的 Docker Compose 也会自动读取 `.env` 文件中的变量。

`.env` 示例：

```dotenv
APP_TAG=mlflow
IMAGE_NAME=fastvit_trainer
APP_CMD="python3 train.py --config configs/base_config_fastvit_sa12.apple_in1k.yaml"
```

各变量说明：

- `IMAGE_NAME`：`fastvit_trainer` 服务使用的 Docker 镜像名称。
- `APP_TAG`：与 `IMAGE_NAME` 组合使用的镜像标签。
- `APP_CMD`：传入容器执行的训练命令。

常见工作流程：

```bash
# 1. 修改 .env
# 2. 启动训练栈
./run_training_stack.sh

# PowerShell
.\run_training_stack.ps1
```

例如，如果你想在不修改 `docker-compose.yml` 的情况下切换配置文件，可以直接修改 `.env` 中的 `APP_CMD`：

```dotenv
APP_CMD="python3 train.py --config configs/base_config.yaml --run-name experiment_a"
```

你也可以通过以下命令检查变量解析后的 Compose 配置：

```bash
docker compose config
```

#### 本地安装
```bash
pip install -r requirements.txt
```

#### 启用 NVIDIA DALI（Python 3.8）
安装与 CUDA 运行时匹配的 DALI wheel（DALI 1.38 支持 Python 3.8）：

```bash
pip install nvidia-dali-cuda120==1.38.0
```

如果你的 CUDA 版本不是 12.x，请将 `cuda120` 替换为对应版本的 wheel。

### 3. 训练
根据需要修改 `configs/base_config.yaml`，然后执行：

```bash
python train.py --config configs/base_config.yaml
```

### 4. 推理
```bash
python infer.py --config configs/base_config.yaml --checkpoint output/model_best.pth.tar --image test.jpg
```

## 配置说明

`configs/base_config.yaml` 中的关键参数：

- `model_name`：选择模型，如 `fastvit_t8`、`fastvit_ma36` 等。
- `lr`：学习率，默认值为 `0.001`。
- `batch_size`：每张 GPU 的 batch size。
- `use_amp`：启用或关闭混合精度训练。
- `use_dali`：启用 NVIDIA DALI 数据加载路径。
- `val_resize_size`：验证集在中心裁剪前的短边缩放尺寸，默认值为 `256`。

### MLflow

在配置文件中启用 MLflow 以跟踪训练元数据：

```yaml
mlflow:
    enabled: true
    experiment_name: fastvit-trainer
    tracking_uri: http://127.0.0.1:5000
    log_model: true
    log_checkpoints: false
    system_metrics: false
    register_model: false
    registered_model_name: ''   # 为空时默认使用经过净化的 model_name
    model_stage: 'Staging'      # 设为 '' 可跳过阶段转换
    tags:
        project: fastvit
        dataset: RealWaste
```

启用后，训练器会记录：

- 展平后的配置参数
- 每个 epoch 的训练与验证指标
- 复制后的运行配置文件和 `train.log`
- 当 `log_model: true` 时记录最优 checkpoint 的 PyTorch 模型产物
- 当 `log_checkpoints: true` 时记录 checkpoint 产物

#### 模型注册表

将 `register_model` 设为 `true`（需同时开启 `log_model: true`），可在训练结束后自动将最优模型注册到 MLflow 模型注册表：

- 训练结束时加载验证精度最高的最优模型权重。
- 通过 `mlflow.pytorch.log_model()` 并设置 `registered_model_name` 参数记录模型，使其出现在 MLflow UI 的 **Models** 页面。
- 若 `model_stage` 不为空（默认为 `Staging`），将通过 `MlflowClient` 将注册版本转换到对应阶段。
- 训练日志中会打印 run ID 和已注册的模型版本，方便追踪。

#### 系统指标

设置 `system_metrics: true` 可通过 MLflow 系统指标功能自动追踪训练期间的硬件资源利用情况，包括：

- CPU 利用率
- 内存使用量
- GPU 利用率与显存占用
- GPU 功耗与温度
- 磁盘使用量与网络 I/O

GPU 指标需要安装 `pynvml` 包及 NVIDIA 驱动。系统指标将以 `system/cpu_utilization_percentage`、`system/system_memory_usage_megabytes`、`system/gpu_utilization_percentage` 等形式记录，并可在 MLflow UI 中与训练指标一同查看。

如有需要，可通过以下命令启动本地 MLflow 服务：

```bash
mlflow server --host 0.0.0.0 --port 5000
```

## 致谢
- [timm](https://github.com/huggingface/pytorch-image-models)
- [FastViT Paper](https://arxiv.org/abs/2303.14189)