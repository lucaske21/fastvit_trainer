import os
import torch
import yaml
import logging
import queue
import threading
from torch.utils.tensorboard import SummaryWriter


def _flatten_config(config, prefix=''):
    flat_config = {}
    for key, value in config.items():
        composed_key = f'{prefix}.{key}' if prefix else str(key)
        if isinstance(value, dict):
            flat_config.update(_flatten_config(value, composed_key))
        else:
            flat_config[composed_key] = value
    return flat_config


def _stringify_param(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class MLflowTracker:
    def __init__(self, config, run_dir, logger, run_name=None, resume_path=None):
        mlflow_config = config.get('mlflow', {}) or {}
        self.enabled = bool(mlflow_config.get('enabled', False))
        self.run_dir = run_dir
        self.logger = logger
        self.run_name = run_name
        self.resume_path = resume_path
        self.mlflow = None
        self.active_run = None
        self.artifact_paths = set()
        self.log_model = bool(mlflow_config.get('log_model', False))
        self.log_checkpoints = bool(mlflow_config.get('log_checkpoints', False))
        self.tags = mlflow_config.get('tags', {}) or {}
        self.experiment_name = mlflow_config.get('experiment_name', 'fastvit-trainer')
        self.tracking_uri = mlflow_config.get('tracking_uri')

    def start(self, config):
        if not self.enabled:
            return

        try:
            import mlflow
        except ImportError as exc:
            raise ImportError(
                'MLflow support is enabled in config, but mlflow is not installed. '
                'Install it with `pip install mlflow` or disable config.mlflow.enabled.'
            ) from exc

        self.mlflow = mlflow

        if self.tracking_uri:
            self.mlflow.set_tracking_uri(self.tracking_uri)
        self.mlflow.set_experiment(self.experiment_name)

        run_kwargs = {}
        if self.run_name:
            run_kwargs['run_name'] = self.run_name

        self.active_run = self.mlflow.start_run(**run_kwargs)
        self.mlflow.set_tag('model_name', config.get('model_name', 'unknown'))
        self.mlflow.set_tag('run_dir', os.path.abspath(self.run_dir))
        if self.resume_path:
            self.mlflow.set_tag('resume_checkpoint', os.path.abspath(self.resume_path))
        for key, value in self.tags.items():
            self.mlflow.set_tag(key, value)

        params = {
            key: _stringify_param(value)
            for key, value in _flatten_config(config).items()
            if not key.startswith('mlflow.tags.')
        }
        self.mlflow.log_params(params)
        self.logger.info(
            'MLflow tracking enabled: experiment=%s, run_id=%s',
            self.experiment_name,
            self.active_run.info.run_id,
        )

    def log_metrics(self, metrics, step):
        if not self.enabled or self.mlflow is None:
            return
        self.mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path, artifact_path=None):
        if not self.enabled or self.mlflow is None:
            return
        normalized_path = os.path.abspath(path)
        if not os.path.exists(normalized_path) or normalized_path in self.artifact_paths:
            return
        self.mlflow.log_artifact(normalized_path, artifact_path=artifact_path)
        self.artifact_paths.add(normalized_path)

    def log_model_artifact(self, model):
        if not self.enabled or self.mlflow is None or not self.log_model:
            return
        self.mlflow.pytorch.log_model(model, artifact_path='model')

    def finish(self, status='FINISHED'):
        if not self.enabled or self.mlflow is None:
            return
        self.mlflow.end_run(status=status)
        self.active_run = None

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger('FastViT-Trainer')
    logger.setLevel(logging.INFO)
    # Clear any handlers added by a previous run in the same process
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(os.path.join(log_dir, 'train.log'))
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth.tar'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(state, os.path.join(checkpoint_dir, filename))
    if is_best:
        torch.save(state, os.path.join(checkpoint_dir, 'model_best.pth.tar'))


class AsyncCheckpointSaver:
    """Background checkpoint writer to reduce epoch-end blocking on disk I/O."""
    def __init__(self, checkpoint_dir, filename='checkpoint.pth.tar', max_queue_size=2):
        self.checkpoint_dir = checkpoint_dir
        self.filename = filename
        self._queue = queue.Queue(maxsize=max_queue_size)
        self._error = None
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    self._queue.task_done()
                    break
                state, is_best = item
                save_checkpoint(state, is_best, self.checkpoint_dir, self.filename)
                self._queue.task_done()
        except Exception as exc:
            self._error = exc

    def submit(self, state, is_best):
        if self._error is not None:
            raise self._error
        self._queue.put((state, is_best))

    def close(self):
        if self._error is not None:
            raise self._error
        self._queue.put(None)
        self._thread.join()
        if self._error is not None:
            raise self._error

def get_optimizer_scheduler(model, config):
    """
    基於 timm 推薦配置。
    """
    from timm.optim import create_optimizer_v2
    from timm.scheduler import create_scheduler_v2

    optimizer = create_optimizer_v2(
        model,
        opt=config.get('opt', 'adamw'),
        lr=config.get('lr', 1e-3),
        weight_decay=config.get('weight_decay', 0.05),
        momentum=config.get('momentum', 0.9)
    )

    scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        sched=config.get('sched', 'cosine'),
        num_epochs=config.get('epochs', 300),
        decay_epochs=config.get('decay_epochs', 30),
        decay_rate=config.get('decay_rate', 0.1),
        warmup_lr=config.get('warmup_lr', 1e-6),
        warmup_epochs=config.get('warmup_epochs', 5),
        cooldown_epochs=config.get('cooldown_epochs', 10),
        min_lr=config.get('min_lr', 1e-5),
        cycle_mul=config.get('cycle_mul', 1.0),
        cycle_decay=config.get('cycle_decay', 0.1),
        cycle_limit=config.get('cycle_limit', 1),
    )

    return optimizer, scheduler
