# Changelog

## [Unreleased] - 2026-04-09

### Improved — ONNX 导出能力增强

- **`export_onnx.py` — `opset_version` 可配置**：新增 `--opset-version` 参数，默认保持 `11`，可按部署环境切换到更高 ONNX opset。

- **`export_onnx.py` — 类别数自动匹配 checkpoint**：导出时从 checkpoint 自动推断分类头输出维度，避免 `head.fc` 的 shape mismatch（如训练 9 类、配置为 1000 类时）。

- **`export_onnx.py` — ONNX metadata 写入 class names**：导出后自动写入模型元信息（`model_name`、`num_classes`、`opset_version`、`class_names`）。
  - `class_names` 支持来源：
    - `--class-names` 传入逗号分隔字符串
    - `--class-names` 指向 `.json`（list）或 `.txt`（逐行）文件
    - 未显式传入时，自动从 `data_dir/train` 子目录推断

- **`export_onnx.py` — 导出流程稳健性改进**：
  - 导出阶段强制 `pretrained=False`，避免无必要预训练权重下载
  - `torch.load` 兼容 `weights_only=True`（新版本）与旧版本参数差异

## [Unreleased] - 2026-04-08

### Improved — GPU 利用率稳定性与 I/O 并行

- **`datasets/dataloader.py` — 保持相容前提下启用 GPU 预取管线**：在保留 `drop_last` 与 `persistent_workers` 的前提下，采用 `DataLoader + PrefetchLoader` 组合。
  - 使用 `fast_collate` + `PrefetchLoader`，将 normalize 与 random erasing 放到 GPU 端执行
  - `num_workers > 0` 时启用 `prefetch_factor`
  - 无 CUDA 时自动回退到非 GPU prefetch 路径

- **`train.py` — 降低 train/validate 切换导致的空转**：新增验证与保存频率控制。
  - `validate_interval`：按间隔执行验证（默认 5）
  - `val_batch_size`：支持验证阶段使用更大 batch（默认 256）
  - CUDA 下启用 `torch.backends.cudnn.benchmark = True` 提升固定输入尺寸吞吐

- **`utils/utils.py` + `train.py` — 多线程异步 checkpoint 写盘**：新增 `AsyncCheckpointSaver` 后台线程，训练主线程不再每个 epoch 同步阻塞等待磁盘 I/O。
  - 新增 `checkpoint_interval`（默认 5）
  - 在「达到保存间隔 / 产生 best / 最后一个 epoch」时提交异步保存任务


## [Unreleased] - 2026-04-07

### Added — 多次训练支持与输出目录整理

- **`train.py` — 按时间戳隔离每次训练输出**：每次启动训练自动在 `output/` 下创建独立子目录，格式为 `YYYYMMDD_HHMMSS_<model_name>[_<run_name>]/`，内含：
  - `config.yaml` — 本次使用的配置副本，便于复现
  - `train.log` — 训练日志
  - `tensorboard/` — TensorBoard 事件文件
  - `checkpoints/checkpoint.pth.tar` — 最新 checkpoint
  - `checkpoints/model_best.pth.tar` — 最优 checkpoint

- **`train.py` — `--run-name` 参数**：支持为本次训练附加自定义标签，追加至目录名末尾（如 `..._finetune`）。

- **`train.py` — `--resume` 参数**：支持从指定 checkpoint 恢复训练，自动还原模型权重、优化器、调度器状态及已完成的 epoch 数。

- **`utils/utils.py` — `setup_logger` 多次调用安全**：改用独立 `Logger` 实例并在创建前清空旧 handler，避免同一进程多次训练时日志重复输出。

- **`docker-compose.yml` — TensorBoard 多 run 支持**：将 `--logdir` 从 `/workspace/output/tensorboard` 改为 `/workspace/output`，TensorBoard 递归扫描所有子目录，自动将每次训练的日志作为独立 run 展示在 UI 中。



### Fixed

- **`timm` 版本兼容性错误**：将 `requirements.txt` 中 `timm` 版本限制为 `>=0.9.0,<1.0.0`。
  `timm>=1.0` 在 `adafactor_bv.py` 中使用了 `tuple[int, int]` 内置泛型语法，仅 Python 3.9+ 支持，容器运行环境为 Python 3.8，导致 `TypeError: 'type' object is not subscriptable`。

- **TensorBoard 无法访问**：原配置仅暴露端口，容器内并无 TensorBoard 进程运行。
  将 TensorBoard 拆分为独立服务，挂载同一 `output/` 卷，使用 `python:3.8-slim` 轻量镜像以节省内存与存储，监听 `0.0.0.0:6006` 对外提供服务。

### Added

- **`.gitignore`**：新增仓库 `.gitignore`，覆盖 Python 缓存、虚拟环境、`data/`、`output/`、模型权重文件、TensorBoard 事件文件、IDE 及操作系统产生的临时文件。

- **`docker-compose.yml` — `ipc: host`**：为训练服务设置 `ipc: host`，避免共享内存不足导致的 DataLoader 多进程问题。

- **ONNX 导出脚本 `export_onnx.py`**：新增模型导出脚本，支持将训练好的 checkpoint 导出为 ONNX 格式（opset 11），特性包括：
  - 在 CPU 上导出，无需 CUDA 依赖
  - 动态 batch 轴（`dynamic_axes`）
  - 常量折叠（`do_constant_folding=True`）
  - 同步在 `requirements.txt` 中新增 `onnx` 依赖

### Fixed — 训练/验证速度瓶颈

| # | 位置 | 问题 | 修复方式 |
|---|------|------|----------|
| 1 | `datasets/dataloader.py` | 缺少 `persistent_workers=True`，每 epoch 重建 worker 进程；`prefetch_factor` 使用默认值 | 切换至 `create_loader` 并设置 `persistent_workers=True` |
| 2 | `engine/trainer.py` | 验证循环每个 batch 重复实例化 `nn.CrossEntropyLoss()` | 改为复用 `self.criterion` |
| 3 | `datasets/dataloader.py` | `use_prefetcher=False` 放弃了 timm CUDA Prefetcher，所有 augment/normalize 在 CPU 完成 | 改用 `create_transform(use_prefetcher=True)` + `create_loader(use_prefetcher=True)`，transform 省略 ToTensor/Normalize，由 `fast_collate` + `PrefetchLoader` 在 GPU 完成；Random Erasing 也移至 GPU 执行 |
| 4 | `engine/trainer.py` | 验证时未开启 `torch.cuda.amp.autocast()`，以 FP32 运行 | 在 `validate()` 的前向传播外包裹 `autocast(enabled=self.scaler is not None)` |
| 5 | `engine/trainer.py` | `optimizer.zero_grad()` 未设置 `set_to_none=True`，多一次不必要的内存写操作 | 修改为 `optimizer.zero_grad(set_to_none=True)` |
