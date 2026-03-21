"""Train EEGNet for single-target regression."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from ...models.dtos import BaseTaskDTO, EEGNetRegTrainParamsDTO
from ...services.ai_models.deep_learning import EEGNetReg
from . import register_dl
from .training_utils import (
    build_lr_scheduler,
    build_dataloaders,
    build_regression_loss,
    compute_regression_metrics,
    eval_loop,
    resolve_device_and_seed,
    resolve_early_stopping_patience,
    split_with_groups,
    train_loop,
)

logger = logging.getLogger(__name__)


@register_dl("Train EEGNet Reg", EEGNetRegTrainParamsDTO)
def train_eegnet_reg(self, task_dto: BaseTaskDTO, params: EEGNetRegTrainParamsDTO):
    """Train a EEGNetReg from a saved DL epoch tensor dataset and return results."""

    selected = self.selected_value(getattr(params, "dataset_path", None))
    task_name = getattr(task_dto, "task", None)
    dataset_path = self.resolve_dataset_path(selected, task_name)
    if not dataset_path:
        return {"status": "no_dataset_selected", "message": "Select a dataset before training."}

    arrays = self.load_xyg_dataset(dataset_path)
    x = np.asarray(arrays["x"], dtype=np.float32)
    y = np.asarray(arrays["y"], dtype=np.float32)
    groups = np.asarray(arrays["group"]).astype(str)

    if x.ndim != 4 or x.shape[1] != 1:
        return {"status": "bad_shape", "message": f"Expected x shape (N,1,C,T), got {x.shape}."}
    if len(x) < 10:
        return {"status": "too_few_samples", "message": f"Only {len(x)} samples - need at least 10."}

    valid_mask = np.isfinite(y)
    if not bool(np.any(valid_mask)):
        return {
            "status": "no_valid_targets",
            "message": "No finite values found in dataset y.",
        }

    x = x[valid_mask]
    y = y[valid_mask]
    groups = groups[valid_mask]

    if len(x) < 10:
        return {
            "status": "too_few_valid_samples",
            "message": f"Only {len(x)} samples remain after filtering invalid y values.",
        }

    # Per-epoch z-score over channel/time.
    x_mean = x.mean(axis=(2, 3), keepdims=True)
    x_std = np.clip(x.std(axis=(2, 3), keepdims=True), 1e-6, None)
    x = (x - x_mean) / x_std

    test_split = float(getattr(params, "test_split", 0.2))
    val_split = float(getattr(params, "val_split", 0.2))
    seed = int(getattr(params, "seed", 42))
    x_tr, y_tr, x_val, y_val, x_te, y_te = split_with_groups(
        x,
        y,
        groups,
        test_split=test_split,
        val_split=val_split,
        seed=seed,
    )

    normalize_target = bool(getattr(params, "normalize_target", True))
    y_mean = float(np.mean(y_tr))
    y_std = max(float(np.std(y_tr)), 1e-8)

    if normalize_target:
        y_tr_fit = ((y_tr - y_mean) / y_std).astype(np.float32)
        y_val_fit = ((y_val - y_mean) / y_std).astype(np.float32)
        y_te_fit = ((y_te - y_mean) / y_std).astype(np.float32)
    else:
        y_tr_fit, y_val_fit, y_te_fit = y_tr, y_val, y_te

    device_str, device = resolve_device_and_seed(
        self.selected_value(getattr(params, "device", "cpu")),
        seed=seed,
    )

    model = EEGNetReg(
        n_channels=int(x_tr.shape[2]),
        n_timepoints=int(x_tr.shape[3]),
    ).to(device)

    loss_choice = self.selected_value(getattr(params, "regression_loss", "huber"))
    loss_name = str(loss_choice or "huber")
    loss_fn = build_regression_loss(loss_name)

    lr = float(getattr(params, "lr", 1e-3))
    min_lr = float(getattr(params, "min_lr", 1e-6))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = build_lr_scheduler(
        optimizer,
        params,
        min_lr=min_lr,
        select_value=self.selected_value,
    )

    epochs_n = int(getattr(params, "epochs_n", 100))
    batch_size = int(getattr(params, "batch_size", 32))
    es_patience = resolve_early_stopping_patience(params)

    tr_dl, val_dl, te_dl = build_dataloaders(
        x_tr,
        y_tr_fit,
        x_val,
        y_val_fit,
        x_te,
        y_te_fit,
        batch_size,
        target_dtype=torch.float32,
    )

    dataset_name = Path(dataset_path).name
    run_name = str(getattr(params, "run_name", "eegnet_reg"))
    model_name = f"{run_name}_{dataset_name}"

    default_wandb_enabled = bool(os.getenv("WANDB_API_KEY"))
    wandb_enabled = bool(getattr(params, "wandb_enabled", default_wandb_enabled))
    wandb_project = str(getattr(params, "wandb_project", os.getenv("WANDB_PROJECT", "ku-final-eegkit-reg")))
    wandb_entity = getattr(params, "wandb_entity", None)
    wandb_tags_raw = getattr(params, "wandb_tags", None)
    wandb_run: Optional[Any] = None
    epoch_logger = None

    if wandb_enabled:
        try:
            import wandb

            init_kwargs: Dict[str, Any] = {
                "project": wandb_project,
                "name": model_name,
                "config": {
                    "dataset_path": dataset_path,
                    "run_name": run_name,
                    "model_name": model_name,
                    "device": device_str,
                    "n_channels": int(x_tr.shape[2]),
                    "n_timepoints": int(x_tr.shape[3]),
                    "train_samples": int(len(x_tr)),
                    "val_samples": int(len(x_val)),
                    "test_samples": int(len(x_te)),
                    "regression_loss": loss_name,
                    "normalize_target": normalize_target,
                    "lr": lr,
                    "min_lr": min_lr,
                    "scheduler_type": self.selected_value(getattr(params, "scheduler_type", "reduce_on_plateau")),
                    "scheduler_mode": self.selected_value(getattr(params, "scheduler_mode", "min")),
                    "scheduler_patience": int(getattr(params, "scheduler_patience", 10)),
                    "scheduler_threshold": float(getattr(params, "scheduler_threshold", 1e-4)),
                    "scheduler_threshold_mode": self.selected_value(
                        getattr(params, "scheduler_threshold_mode", "rel")
                    ),
                    "scheduler_cooldown": int(getattr(params, "scheduler_cooldown", 0)),
                    "early_stopping": bool(getattr(params, "early_stopping", False)),
                    "early_stopping_patience": int(es_patience),
                    "epochs": epochs_n,
                    "batch_size": batch_size,
                    "seed": seed,
                },
            }
            if wandb_entity:
                init_kwargs["entity"] = str(wandb_entity)
            if wandb_tags_raw:
                if isinstance(wandb_tags_raw, str):
                    tags = [t.strip() for t in wandb_tags_raw.split(",") if t.strip()]
                elif isinstance(wandb_tags_raw, (list, tuple)):
                    tags = [str(t).strip() for t in wandb_tags_raw if str(t).strip()]
                else:
                    tags = []
                if tags:
                    init_kwargs["tags"] = tags

            wandb_run = wandb.init(**init_kwargs)
            epoch_logger = lambda payload: wandb_run.log(payload)
            logger.info("W&B run started: %s", getattr(wandb_run, "name", model_name))
        except Exception as exc:
            logger.warning("W&B disabled due to init failure: %s", exc)

    logger.info(
        "Training EEGNet Reg: target=dataset_y device=%s train/val/test=%d/%d/%d",
        device_str,
        len(x_tr),
        len(x_val),
        len(x_te),
    )

    try:
        model, best_epoch = train_loop(
            model,
            tr_dl,
            val_dl,
            loss_fn,
            optimizer,
            scheduler,
            epochs_n,
            device,
            es_patience,
            epoch_logger=epoch_logger,
            classification=False,
        )

        y_true_fit, y_pred_fit = eval_loop(model, te_dl, device, classification=False)
        if normalize_target:
            y_true = (y_true_fit * y_std + y_mean).astype(np.float32)
            y_pred = (y_pred_fit * y_std + y_mean).astype(np.float32)
        else:
            y_true, y_pred = y_true_fit, y_pred_fit

        metrics = compute_regression_metrics(y_true, y_pred)
        logger.info("EEGNet Reg training done. Metrics: %s", metrics)

        if wandb_run is not None:
            try:
                payload: Dict[str, Any] = {
                    "test/rmse": metrics.get("rmse"),
                    "test/mae": metrics.get("mae"),
                    "test/r2": metrics.get("r2"),
                    "test/explained_variance": metrics.get("explained_variance"),
                    "test/pearson_corr": metrics.get("pearson_corr"),
                    "test/y_true_mean": metrics.get("y_true_mean"),
                    "test/y_pred_mean": metrics.get("y_pred_mean"),
                }
                if best_epoch is not None:
                    payload["train/best_epoch"] = float(best_epoch)
                wandb_run.log(payload)
            except Exception as exc:
                logger.warning("W&B metric logging failed: %s", exc)
    finally:
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception as exc:
                logger.warning("W&B finish failed: %s", exc)

    return {
        "model": model,
        "model_name": model_name,
        "evaluation": metrics,
        "plots": {},
        "metadata": {
            "model_name": model_name,
            "dataset_path": dataset_path,
            "target": "dataset_y",
            "n_channels": int(x_tr.shape[2]),
            "n_timepoints": int(x_tr.shape[3]),
            "device": device_str,
            "regression_loss": loss_name,
            "normalize_target": normalize_target,
            "target_mean_train": y_mean,
            "target_std_train": y_std,
            "train_samples": int(len(x_tr)),
            "val_samples": int(len(x_val)),
            "test_samples": int(len(x_te)),
            "run_name": run_name,
            "best_epoch": int(best_epoch) if best_epoch is not None else -1,
            "wandb_enabled": wandb_run is not None,
        },
    }
