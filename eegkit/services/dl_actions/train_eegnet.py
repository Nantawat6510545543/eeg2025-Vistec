"""Train a binary EEGNet on a saved DL epoch tensor dataset."""

from __future__ import annotations

import logging
from pathlib import Path

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


def _load_and_validate_dataset(self, task_dto: BaseTaskDTO, params: DLTrainParamsDTO):
    """Resolve dataset selection, load arrays, and validate shape/min rows."""
    selected = self.selected_value(getattr(params, "dataset_path", None))
    task_name = getattr(task_dto, "task", None)
    dataset_path = self.resolve_dataset_path(selected, task_name)
    if not dataset_path:
        return None, {"status": "no_dataset_selected", "message": "Select a dataset before training."}

    arrays = self.load_xyg_dataset(dataset_path)
    x = np.asarray(arrays["x"], dtype=np.float32)
    y = (np.asarray(arrays["y"]) > 0).astype(np.int64)
    groups = np.asarray(arrays["group"]).astype(str)

    if x.ndim != 4 or x.shape[1] != 1:
        return None, {"status": "bad_shape", "message": f"Expected x shape (N,1,C,T), got {x.shape}."}
    if len(x) < 10:
        return None, {"status": "too_few_samples", "message": f"Only {len(x)} samples — need at least 10."}

    return (dataset_path, x, y, groups), None


@register_dl("Train EEGNet", DLTrainParamsDTO)
def train_eegnet(self, task_dto: BaseTaskDTO, params: DLTrainParamsDTO):
    """Train a binary EEGNet from a saved DL epoch tensor dataset and return results."""
    loaded, error = _load_and_validate_dataset(self, task_dto, params)
    if error is not None:
        return error
    dataset_path, x, y, groups = loaded

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
    logger.info(
        "Training EEGNet: n_ch=%d n_t=%d device=%s train/val/test=%d/%d/%d",
        n_channels, n_timepoints, device_str, len(x_tr), len(x_val), len(x_te),
    )

    model = train_loop(
        model, tr_dl, val_dl, loss_fn, optimizer, scheduler, epochs_n, device, es_patience
    )
    y_true, y_pred, y_prob = eval_loop(model, te_dl, device)
    metrics = compute_binary_metrics(y_true, y_pred, y_prob)
    logger.info("EEGNet training done. Metrics: %s", metrics)

    dataset_name = Path(dataset_path).name
    run_name = str(getattr(params, "run_name", "dl_train"))
    model_name = f"{run_name}_{dataset_name}"
    shaded_error_bar = build_shaded_error_bar_plot(x_te, y_te, model_name)

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
        },
    }
