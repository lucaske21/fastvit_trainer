import argparse
import os
import copy
from datetime import datetime
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import yaml
from torch.utils.tensorboard import SummaryWriter

from datasets.dataloader import get_dataloaders
from engine.trainer import Trainer
from models.builder import create_fastvit_model
from utils.utils import (
    AsyncCheckpointSaver,
    MLflowTracker,
    get_optimizer_scheduler,
    load_config,
    setup_logger,
)


def make_run_dir(base_output_dir: str, model_name: str, run_name: Optional[str]) -> str:
    """Create a unique run directory under base_output_dir."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_model = model_name.replace('/', '_').replace('.', '-')
    folder = f"{timestamp}_{safe_model}"
    if run_name:
        folder = f"{folder}_{run_name}"
    run_dir = os.path.join(base_output_dir, folder)
    os.makedirs(os.path.join(run_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'tensorboard'), exist_ok=True)
    return run_dir


def _dump_run_config(config: Dict[str, Any], run_dir: str) -> str:
    config_path = os.path.join(run_dir, 'config.yaml')
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    return config_path


def _is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return 'out of memory' in str(exc).lower()


def run_training(
    config: Dict[str, Any],
    run_name: Optional[str] = None,
    resume: Optional[str] = None,
    trial: Optional[Any] = None,
) -> Dict[str, Any]:
    run_config = copy.deepcopy(config)

    if torch.cuda.is_available() and run_config.get('device', 'cuda') == 'cuda':
        torch.backends.cudnn.benchmark = True

    run_dir = make_run_dir(run_config['output_dir'], run_config['model_name'], run_name)
    saved_config_path = _dump_run_config(run_config, run_dir)

    logger = setup_logger(run_dir)
    writer = SummaryWriter(log_dir=os.path.join(run_dir, 'tensorboard'))
    logger.info(f"Run directory: {run_dir}")

    mlflow_tracker = MLflowTracker(run_config, run_dir, logger, run_name=run_name, resume_path=resume)

    device = torch.device(run_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    logger.info(f"Using device: {device}")

    train_loader, val_loader, num_classes = get_dataloaders(run_config)
    logger.info(f"Datasets loaded: {num_classes} classes found.")

    model = create_fastvit_model(run_config, num_classes).to(device)
    logger.info(f"Model {run_config['model_name']} created.")

    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = get_optimizer_scheduler(model, run_config)
    trainer = Trainer(model, optimizer, scheduler, criterion, device, run_config)

    best_acc1 = 0.0
    best_acc5 = 0.0
    best_val_loss = float('inf')
    best_epoch = 0
    start_epoch = 1

    if resume:
        if not os.path.isfile(resume):
            raise FileNotFoundError(f"Checkpoint not found: {resume}")
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        best_acc1 = float(checkpoint.get('best_acc1', 0.0))
        start_epoch = int(checkpoint['epoch']) + 1
        logger.info(
            f"Resumed from {resume} "
            f"(epoch {checkpoint['epoch']}, best_acc1={best_acc1:.2f}%)"
        )

    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    checkpoint_saver = AsyncCheckpointSaver(checkpoint_dir)

    validate_interval = max(1, int(run_config.get('validate_interval', 5)))
    checkpoint_interval = max(1, int(run_config.get('checkpoint_interval', validate_interval)))

    status = 'FAILED'
    try:
        mlflow_tracker.start(run_config)
        mlflow_tracker.log_artifact(saved_config_path, artifact_path='configs')

        for epoch in range(start_epoch, int(run_config['epochs']) + 1):
            train_loss, train_acc1, train_acc5 = trainer.train_one_epoch(train_loader, epoch)

            do_validate = (epoch % validate_interval == 0) or (epoch == int(run_config['epochs']))
            if do_validate:
                val_loss, val_acc1, val_acc5 = trainer.validate(val_loader)
            else:
                val_loss, val_acc1, val_acc5 = float('nan'), float('nan'), float('nan')

            if do_validate:
                logger.info(
                    f"Epoch {epoch}/{run_config['epochs']}: "
                    f"Train Loss={train_loss:.4f} Acc@1={train_acc1:.2f}% | "
                    f"Val Loss={val_loss:.4f} Acc@1={val_acc1:.2f}%"
                )
            else:
                logger.info(
                    f"Epoch {epoch}/{run_config['epochs']}: "
                    f"Train Loss={train_loss:.4f} Acc@1={train_acc1:.2f}% | "
                    f"Val skipped (validate_interval={validate_interval})"
                )

            writer.add_scalar('Loss/train', train_loss, epoch)
            if do_validate:
                writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc1, epoch)
            if do_validate:
                writer.add_scalar('Accuracy/val', val_acc1, epoch)

            current_lr = optimizer.param_groups[0]['lr'] if optimizer.param_groups else 0.0
            metric_payload = {
                'train_loss': train_loss,
                'train_acc1': train_acc1,
                'train_acc5': train_acc5,
                'lr': current_lr,
            }
            if do_validate:
                metric_payload.update({
                    'val_loss': val_loss,
                    'val_acc1': val_acc1,
                    'val_acc5': val_acc5,
                })
            mlflow_tracker.log_metrics(metric_payload, step=epoch)

            is_best = do_validate and (val_acc1 > best_acc1)
            if do_validate and is_best:
                best_acc1 = float(val_acc1)
                best_acc5 = float(val_acc5)
                best_val_loss = float(val_loss)
                best_epoch = int(epoch)

            should_save = is_best or (epoch % checkpoint_interval == 0) or (epoch == int(run_config['epochs']))
            if should_save:
                checkpoint_state = {
                    'epoch': epoch,
                    'model_name': run_config['model_name'],
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }
                checkpoint_saver.submit(checkpoint_state, is_best)

            if do_validate and trial is not None:
                trial.report(float(val_acc1), step=epoch)
                if trial.should_prune():
                    status = 'KILLED'
                    try:
                        import optuna
                    except ImportError as exc:
                        raise RuntimeError('Optuna is required when using trial-based pruning.') from exc
                    raise optuna.TrialPruned(f'Pruned at epoch {epoch}')

        mlflow_tracker.log_model_artifact(model)

        # Register the best model to MLflow Model Registry
        best_checkpoint_path = os.path.join(checkpoint_dir, 'model_best.pth.tar')
        if os.path.isfile(best_checkpoint_path):
            logger.info(f'Loading best checkpoint for model registration: {best_checkpoint_path}')
            best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
            model.load_state_dict(best_checkpoint['state_dict'])

            best_metrics = {
                'best_val_acc1': best_acc1,
                'best_val_acc5': best_acc5,
                'best_val_loss': best_val_loss if best_val_loss != float('inf') else float('nan'),
                'best_epoch': best_epoch,
            }

            registration_info = mlflow_tracker.register_best_model(model, best_metrics=best_metrics)
            if registration_info:
                logger.info(
                    f"Model registration complete: "
                    f"run_id={registration_info['run_id']}, "
                    f"version={registration_info['model_version']}"
                )
        else:
            logger.warning(f'Best checkpoint not found at {best_checkpoint_path}, skipping model registration')

        status = 'FINISHED'
    except RuntimeError as exc:
        if trial is not None and _is_oom_error(exc):
            logger.warning('OOM detected during trial run, marking this trial as pruned.')
        raise
    finally:
        writer.close()
        checkpoint_saver.close()
        if mlflow_tracker.log_checkpoints and os.path.isdir(checkpoint_dir):
            for checkpoint_name in sorted(os.listdir(checkpoint_dir)):
                checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
                if os.path.isfile(checkpoint_path):
                    mlflow_tracker.log_artifact(checkpoint_path, artifact_path='checkpoints')
        train_log_path = os.path.join(run_dir, 'train.log')
        if os.path.isfile(train_log_path):
            mlflow_tracker.log_artifact(train_log_path, artifact_path='logs')

        # Capture run_id before finishing the run
        run_id = None
        if mlflow_tracker.enabled and mlflow_tracker.active_run:
            run_id = mlflow_tracker.active_run.info.run_id

        mlflow_tracker.finish(status=status)

    logger.info(f"Training completed. Best Val Acc@1: {best_acc1:.2f}%")
    logger.info(f"Checkpoints saved to: {checkpoint_dir}")
    if run_id:
        logger.info(f"MLflow Run ID: {run_id}")

    return {
        'best_acc1': float(best_acc1),
        'best_acc5': float(best_acc5),
        'best_val_loss': float(best_val_loss) if best_val_loss != float('inf') else float('nan'),
        'best_epoch': int(best_epoch),
        'run_dir': run_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='FastViT Training Framework')
    parser.add_argument('--config', default='configs/base_config.yaml', type=str, help='Path to config file')
    parser.add_argument('--run-name', default=None, type=str, help='Optional tag appended to the run directory name')
    parser.add_argument('--resume', default=None, type=str, help='Path to a checkpoint to resume training from')
    args = parser.parse_args()

    config = load_config(args.config)
    run_training(config, run_name=args.run_name, resume=args.resume)


if __name__ == '__main__':
    main()
