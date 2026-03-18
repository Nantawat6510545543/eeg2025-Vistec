"""Train a binary EEGNet on a saved DL epoch tensor dataset."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from ...models.dtos import BaseTaskDTO, DLTrainParamsDTO
from ...services.ai_models.deep_learning.EEGNetBinary import EEGNetBinary
from .training_utils import (
    build_dataloaders,
    build_loss_function,
    build_shaded_error_bar_plot,
    class_weights,
    compute_binary_metrics,
    eval_loop,
    resolve_device_and_seed,
    split_with_groups,
    train_loop,
)
from . import register_dl

logger = logging.getLogger(__name__)


@register_dl("Train EEGNet", DLTrainParamsDTO)
def train_eegnet(self, task_dto: BaseTaskDTO, params: DLTrainParamsDTO):
    """Train a binary EEGNet from a saved DL epoch tensor dataset and return results."""
    selected = self.selected_value(getattr(params, "dataset_path", None))
    task_name = getattr(task_dto, "task", None)
    dataset_path = self.resolve_dataset_path(selected, task_name)
    if not dataset_path:
        return {"status": "no_dataset_selected", "message": "Select a dataset before training."}

    arrays = self.load_xyg_dataset(dataset_path)
    x = np.asarray(arrays["x"], dtype=np.float32)
    y = (np.asarray(arrays["y"]) > 0).astype(np.int64)
    groups = np.asarray(arrays["group"]).astype(str)

    if x.ndim != 4 or x.shape[1] != 1:
        return {"status": "bad_shape", "message": f"Expected x shape (N,1,C,T), got {x.shape}."}
    if len(x) < 10:
        return {"status": "too_few_samples", "message": f"Only {len(x)} samples — need at least 10."}
    if not np.isfinite(x).all():
        return {"status": "bad_values", "message": "Input dataset contains NaN or Inf values."}
    if np.unique(y).size < 2:
        label_counts = dict(zip(*np.unique(y, return_counts=True)))
        return {
            "status": "single_class_dataset",
            "message": f"Dataset has only one class: {label_counts}. Check CCD label extraction.",
        }

    x_mean = x.mean(axis=(2, 3), keepdims=True)
    x_std = x.std(axis=(2, 3), keepdims=True)
    x = (x - x_mean) / np.clip(x_std, 1e-6, None)

    test_split = float(getattr(params, "test_split", 0.25))
    val_split = float(getattr(params, "val_split", 0.25))
    x_tr, y_tr, x_val, y_val, x_te, y_te = split_with_groups(
        x,
        y,
        groups,
        test_split=test_split,
        val_split=val_split,
        seed=int(getattr(params, "seed", 42)),
    )
    if np.unique(y_tr).size < 2:
        train_counts = dict(zip(*np.unique(y_tr, return_counts=True)))
        return {
            "status": "single_class_train_split",
            "message": f"Training split has only one class: {train_counts}. Adjust split or inspect labels.",
        }

    device_str, device = resolve_device_and_seed(
        self.selected_value(getattr(params, "device", "cpu")),
        seed=int(getattr(params, "seed", 42)),
    )

    n_channels = x_tr.shape[2]
    n_timepoints = x_tr.shape[3]
    model = EEGNetBinary(n_channels=n_channels, n_timepoints=n_timepoints).to(device)

    weighted_sampler = bool(getattr(params, "weighted_sampler", False))
    use_class_weights = bool(getattr(params, "weight_classes", True))
    if weighted_sampler and use_class_weights:
        logger.warning(
            "Both weighted_sampler and weight_classes are enabled; "
            "disabling class-weighted loss to avoid double compensation."
        )
        use_class_weights = False

    weight_tensor = class_weights(y_tr, device) if use_class_weights else None
    loss_choice = self.selected_value(getattr(params, "loss_function", "cross_entropy"))
    loss_name = str(loss_choice or "cross_entropy")
    label_smoothing = float(getattr(params, "label_smoothing", 0.0))
    loss_fn = build_loss_function(loss_name, weight_tensor, label_smoothing)

    lr = float(getattr(params, "lr", 1e-3))
    min_lr = float(getattr(params, "min_lr", 1e-6))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        "min",
        factor=float(getattr(params, "lr_factor", 0.5)),
        patience=int(getattr(params, "patience", 10)),
        min_lr=min_lr,
    )

    epochs_n = int(getattr(params, "epochs_n", 50))
    batch_size = int(getattr(params, "batch_size", 32))
    early_stopping = bool(getattr(params, "early_stopping", False))
    es_patience = int(getattr(params, "patience", 10)) * 2 if early_stopping else 0

    tr_dl, val_dl, te_dl = build_dataloaders(
        x_tr, y_tr, x_val, y_val, x_te, y_te, batch_size, weighted_sampler
    )

    dataset_name = Path(dataset_path).name
    run_name = str(getattr(params, "run_name", "dl_train"))
    model_name = f"{run_name}_{dataset_name}"

    default_wandb_enabled = bool(os.getenv("WANDB_API_KEY"))
    wandb_enabled = bool(getattr(params, "wandb_enabled", default_wandb_enabled))
    wandb_project = str(getattr(params, "wandb_project", os.getenv("WANDB_PROJECT", "eeg2025-vistec")))
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
                    "n_channels": int(n_channels),
                    "n_timepoints": int(n_timepoints),
                    "train_samples": int(len(x_tr)),
                    "val_samples": int(len(x_val)),
                    "test_samples": int(len(x_te)),
                    "loss_function": loss_name,
                    "label_smoothing": label_smoothing,
                    "weighted_sampler": weighted_sampler,
                    "weight_classes": use_class_weights,
                    "lr": lr,
                    "min_lr": min_lr,
                    "epochs": epochs_n,
                    "batch_size": batch_size,
                    "seed": int(getattr(params, "seed", 42)),
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
        "Training EEGNet: n_ch=%d n_t=%d device=%s train/val/test=%d/%d/%d",
        n_channels, n_timepoints, device_str, len(x_tr), len(x_val), len(x_te),
    )
    logger.info(
        "Label balance all/train/val/test: %s / %s / %s / %s",
        dict(zip(*np.unique(y, return_counts=True))),
        dict(zip(*np.unique(y_tr, return_counts=True))),
        dict(zip(*np.unique(y_val, return_counts=True))),
        dict(zip(*np.unique(y_te, return_counts=True))),
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
        )
        y_true, y_pred, y_prob = eval_loop(model, te_dl, device)
        metrics = compute_binary_metrics(y_true, y_pred, y_prob)
        logger.info("EEGNet training done. Metrics: %s", metrics)

        shaded_error_bar = build_shaded_error_bar_plot(x_te, y_te, model_name)

        if wandb_run is not None:
            try:
                payload: Dict[str, Any] = {
                    "test/accuracy": metrics.get("accuracy"),
                    "test/balanced_accuracy": metrics.get("balanced_accuracy"),
                    "test/f1_binary": metrics.get("f1_binary"),
                    "test/f1_macro": metrics.get("f1_macro"),
                    "test/f1_weighted": metrics.get("f1_weighted"),
                    "test/sensitivity": metrics.get("sensitivity"),
                    "test/specificity": metrics.get("specificity"),
                    "test/roc_auc": metrics.get("roc_auc"),
                    "test/pr_auc": metrics.get("pr_auc"),
                }
                cm = metrics.get("confusion_matrix", {})
                if isinstance(cm, dict):
                    payload["test/cm_tn"] = cm.get("tn")
                    payload["test/cm_fp"] = cm.get("fp")
                    payload["test/cm_fn"] = cm.get("fn")
                    payload["test/cm_tp"] = cm.get("tp")
                class_recall = metrics.get("class_recall", {})
                if isinstance(class_recall, dict):
                    payload["test/recall_class_0"] = class_recall.get("0")
                    payload["test/recall_class_1"] = class_recall.get("1")
                class_precision = metrics.get("class_precision", {})
                if isinstance(class_precision, dict):
                    payload["test/precision_class_0"] = class_precision.get("0")
                    payload["test/precision_class_1"] = class_precision.get("1")
                
                payload["train/best_epoch"] = float(best_epoch)

                wandb_run.log(payload)

                if shaded_error_bar is not None:
                    import wandb

                    wandb_run.log({"plots/shaded_error_bar": wandb.Image(shaded_error_bar)})
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
        "plots": {
            "shaded_error_bar": shaded_error_bar,
        },
        "metadata": {
            "model_name": model_name,
            "dataset_path": dataset_path,
            "n_channels": n_channels,
            "n_timepoints": n_timepoints,
            "device": device_str,
            "loss_function": loss_name,
            "label_smoothing": label_smoothing,
            "weight_classes": use_class_weights,
            "weighted_sampler": weighted_sampler,
            "train_samples": int(len(x_tr)),
            "val_samples": int(len(x_val)),
            "test_samples": int(len(x_te)),
            "run_name": run_name,
            "wandb_enabled": wandb_run is not None,
        },
    }
