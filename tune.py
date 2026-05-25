import argparse
import csv
import json
import os
import traceback
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional

import optuna
import torch
import yaml

from train import run_training
from utils.utils import load_config


def _safe_yaml_dump(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def _load_tune_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    return load_config(path) or {}


def _suggest_trial_params(trial: optuna.trial.Trial, tune_config: Dict[str, Any]) -> Dict[str, Any]:
    space = tune_config.get('search_space', {}) or {}

    lr_cfg = space.get('lr', {'low': 1e-5, 'high': 5e-3, 'log': True})
    wd_cfg = space.get('weight_decay', {'low': 1e-4, 'high': 1e-1, 'log': True})
    bs_cfg = space.get('batch_size', {'choices': [64, 128, 256]})
    dpr_cfg = space.get('drop_path_rate', {'low': 0.0, 'high': 0.3})
    warmup_cfg = space.get('warmup_epochs', {'low': 0, 'high': 10})

    return {
        'lr': trial.suggest_float('lr', float(lr_cfg['low']), float(lr_cfg['high']), log=bool(lr_cfg.get('log', True))),
        'weight_decay': trial.suggest_float(
            'weight_decay',
            float(wd_cfg['low']),
            float(wd_cfg['high']),
            log=bool(wd_cfg.get('log', True)),
        ),
        'batch_size': int(trial.suggest_categorical('batch_size', list(bs_cfg.get('choices', [64, 128, 256])))),
        'drop_path_rate': trial.suggest_float(
            'drop_path_rate',
            float(dpr_cfg['low']),
            float(dpr_cfg['high']),
        ),
        'warmup_epochs': int(trial.suggest_int('warmup_epochs', int(warmup_cfg['low']), int(warmup_cfg['high']))),
    }


def _build_trial_config(
    base_config: Dict[str, Any],
    params: Dict[str, Any],
    epochs_override: Optional[int],
) -> Dict[str, Any]:
    trial_config = deepcopy(base_config)
    trial_config.update(params)

    if 'val_batch_size' not in trial_config:
        trial_config['val_batch_size'] = trial_config['batch_size']

    if epochs_override is not None:
        trial_config['epochs'] = int(epochs_override)

    return trial_config


def _serialize_trial(trial: optuna.trial.FrozenTrial) -> Dict[str, Any]:
    return {
        'number': trial.number,
        'state': trial.state.name,
        'value': trial.value,
        'params': trial.params,
        'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
        'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None,
        'duration_seconds': trial.duration.total_seconds() if trial.duration else None,
        'user_attrs': trial.user_attrs,
    }


def _save_study_outputs(study: optuna.study.Study, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    best = {
        'best_trial_number': study.best_trial.number,
        'best_value': study.best_value,
        'best_params': study.best_params,
    }
    with open(os.path.join(output_dir, 'best_result.json'), 'w', encoding='utf-8') as f:
        json.dump(best, f, indent=2, ensure_ascii=False)

    trials = [_serialize_trial(t) for t in study.trials]
    with open(os.path.join(output_dir, 'trials_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(trials, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(output_dir, 'trials_summary.csv')
    fieldnames = ['number', 'state', 'value', 'duration_seconds', 'params', 'user_attrs']
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in trials:
            writer.writerow({
                'number': row['number'],
                'state': row['state'],
                'value': row['value'],
                'duration_seconds': row['duration_seconds'],
                'params': json.dumps(row['params'], ensure_ascii=False),
                'user_attrs': json.dumps(row['user_attrs'], ensure_ascii=False),
            })


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Optuna tuning for FastViT trainer')
    parser.add_argument('--base-config', type=str, default='configs/base_config.yaml', help='Path to base train config')
    parser.add_argument('--tune-config', type=str, default='configs/tune_config.yaml', help='Path to tune config')
    parser.add_argument('--n-trials', type=int, default=20, help='Number of trials')
    parser.add_argument('--timeout', type=int, default=None, help='Maximum tuning time in seconds')
    parser.add_argument('--epochs', type=int, default=None, help='Override epoch count during tuning')
    parser.add_argument('--storage', type=str, default=None, help='Optuna storage URL, e.g. sqlite:///output/optuna.db')
    parser.add_argument('--study-name', type=str, default='fastvit_tune', help='Optuna study name')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to save tune results')
    return parser


def main() -> None:
    args = build_parser().parse_args()

    base_config = load_config(args.base_config)
    tune_config = _load_tune_config(args.tune_config)

    tune_output_dir = args.output_dir or tune_config.get('output_dir')
    if not tune_output_dir:
        tune_output_dir = os.path.join(base_config.get('output_dir', './output'), 'tune')

    os.makedirs(tune_output_dir, exist_ok=True)

    snapshot_path = os.path.join(tune_output_dir, f"tune_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml")
    _safe_yaml_dump(snapshot_path, {
        'base_config_path': args.base_config,
        'tune_config_path': args.tune_config,
        'n_trials': args.n_trials,
        'timeout': args.timeout,
        'epochs': args.epochs,
        'study_name': args.study_name,
        'storage': args.storage,
    })

    pruner_cfg = tune_config.get('pruner', {}) or {}
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=int(pruner_cfg.get('n_startup_trials', 3)),
        n_warmup_steps=int(pruner_cfg.get('n_warmup_steps', 1)),
    )

    study = optuna.create_study(
        study_name=args.study_name,
        direction='maximize',
        storage=args.storage,
        load_if_exists=True,
        pruner=pruner,
    )

    def objective(trial: optuna.trial.Trial) -> float:
        params = _suggest_trial_params(trial, tune_config)
        trial_config = _build_trial_config(base_config, params, args.epochs)

        trial.set_user_attr('params', params)
        print(f"[Trial {trial.number}] Start with params: {json.dumps(params, ensure_ascii=False)}")

        try:
            result = run_training(
                trial_config,
                run_name=f"trial_{trial.number}",
                resume=None,
                trial=trial,
            )
            target = float(result['best_acc1'])
            trial.set_user_attr('run_dir', result['run_dir'])
            trial.set_user_attr('best_epoch', int(result['best_epoch']))
            trial.set_user_attr('best_acc5', float(result['best_acc5']))
            trial.set_user_attr('best_val_loss', float(result['best_val_loss']))
            print(
                f"[Trial {trial.number}] Done: "
                f"best_acc1={target:.4f}, best_epoch={result['best_epoch']}, run_dir={result['run_dir']}"
            )
            return target
        except RuntimeError as exc:
            if isinstance(exc, torch.cuda.OutOfMemoryError) or 'out of memory' in str(exc).lower():
                print(f"[Trial {trial.number}] Pruned due to runtime condition: {exc}")
                raise optuna.TrialPruned(str(exc)) from exc
            print(f"[Trial {trial.number}] RuntimeError: {exc}")
            raise
        except optuna.TrialPruned:
            raise
        except Exception as exc:
            print(f"[Trial {trial.number}] Failed: {exc}")
            traceback.print_exc()
            raise

    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        catch=(Exception,),
    )

    if len(study.trials) == 0:
        raise RuntimeError('No trials were executed.')

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        raise RuntimeError('No trial completed successfully; cannot determine best trial.')

    _save_study_outputs(study, tune_output_dir)

    print('\n=== Optuna Tune Summary ===')
    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best Value (best_acc1): {study.best_value:.6f}")
    print(f"Best Params: {json.dumps(study.best_params, ensure_ascii=False)}")
    print(f"Saved summary to: {tune_output_dir}")


if __name__ == '__main__':
    main()
