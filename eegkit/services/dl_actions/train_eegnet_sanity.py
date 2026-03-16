"""Sanity-check whether the standard EEGNet can overfit a tiny balanced subset."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from ...models.dtos import BaseTaskDTO, DLTrainParamsDTO
from ...services.ai_models.deep_learning.EEGNetBinary import EEGNetBinary
from . import register_dl
from .training_utils import (
    build_dataloaders,
    build_loss_function,
    build_shaded_error_bar_plot,
    compute_binary_metrics,
    eval_loop,
    resolve_device_and_seed,
    train_loop,
)

logger = logging.getLogger(__name__)


def _pick_balanced_subset(
    x: np.ndarray,
    y: np.ndarray,
    subset_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select a small balanced subset for a memorization sanity check."""
    classes, counts = np.unique(y, return_counts=True)
    if classes.size < 2:
        raise ValueError(f"Need two classes for sanity check, got {dict(zip(classes, counts))}")

    per_class = min(max(subset_size // classes.size, 1), int(counts.min()))
    rng = np.random.default_rng(seed)
    indices = []
    for cls in classes:
        cls_idx = np.flatnonzero(y == cls)
        rng.shuffle(cls_idx)
        indices.extend(cls_idx[:per_class].tolist())

    rng.shuffle(indices)
    idx = np.asarray(indices, dtype=np.int64)
    return x[idx], y[idx]


@register_dl("Sanity Check EEGNet", DLTrainParamsDTO)
def train_eegnet_sanity(self, task_dto: BaseTaskDTO, params: DLTrainParamsDTO):
    """Overfit a tiny balanced subset to verify the standard EEGNet can learn."""
    selected = self.selected_value(getattr(params, "dataset_path", None))
    task_name = getattr(task_dto, "task", None)
    dataset_path = self.resolve_dataset_path(selected, task_name)
    if not dataset_path:
        return {"status": "no_dataset_selected", "message": "Select a dataset before training."}

    arrays = self.load_xyg_dataset(dataset_path)
    x = np.asarray(arrays["x"], dtype=np.float32)
    y = (np.asarray(arrays["y"]) > 0).astype(np.int64)

    if x.ndim != 4 or x.shape[1] != 1:
        return {"status": "bad_shape", "message": f"Expected x shape (N,1,C,T), got {x.shape}."}
    if len(x) < 4:
        return {"status": "too_few_samples", "message": f"Only {len(x)} samples - need at least 4."}
    if not np.isfinite(x).all():
        return {"status": "bad_values", "message": "Input dataset contains NaN or Inf values."}
    if np.unique(y).size < 2:
        label_counts = dict(zip(*np.unique(y, return_counts=True)))
        return {
            "status": "single_class_dataset",
            "message": f"Dataset has only one class: {label_counts}. Check CCD label extraction.",
        }

    x_mean = x.mean(axis=(2, 3), keepdims=True)
    x_std = np.clip(x.std(axis=(2, 3), keepdims=True), 1e-6, None)
    x = (x - x_mean) / x_std

    seed = int(getattr(params, "seed", 42))
    requested_subset = int(getattr(params, "batch_size", 32))
    subset_size = min(max(requested_subset, 4), 32, len(y))
    x_small, y_small = _pick_balanced_subset(x, y, subset_size, seed)

    device_str, device = resolve_device_and_seed(
        self.selected_value(getattr(params, "device", "cpu")),
        seed=seed,
    )

    n_channels = x_small.shape[2]
    n_timepoints = x_small.shape[3]
    model = EEGNetBinary(
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        dropout_rate=0.0,
    ).to(device)

    loss_name = "cross_entropy"
    label_smoothing = 0.0
    loss_fn = build_loss_function(loss_name, None, label_smoothing)

    lr = float(getattr(params, "lr", 1e-3))
    min_lr = float(getattr(params, "min_lr", 1e-6))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        "min",
        factor=float(getattr(params, "lr_factor", 0.5)),
        patience=max(5, int(getattr(params, "patience", 10)) // 2),
        min_lr=min_lr,
    )

    epochs_n = max(int(getattr(params, "epochs_n", 50)), 100)
    batch_size = min(int(getattr(params, "batch_size", 32)), len(x_small))

    tr_dl, val_dl, te_dl = build_dataloaders(
        x_small,
        y_small,
        x_small,
        y_small,
        x_small,
        y_small,
        batch_size,
        False,
    )
    logger.info(
        "Sanity-check EEGNet: n_ch=%d n_t=%d device=%s subset=%d epochs=%d",
        n_channels,
        n_timepoints,
        device_str,
        len(x_small),
        epochs_n,
    )
    logger.info("Label balance (sanity subset): %s", dict(zip(*np.unique(y_small, return_counts=True))))

    model = train_loop(tr_model := model, tr_dl, val_dl, loss_fn, optimizer, scheduler, epochs_n, device, 0)
    y_true, y_pred, y_prob = eval_loop(model, te_dl, device)
    metrics = compute_binary_metrics(y_true, y_pred, y_prob)
    logger.info("EEGNet sanity-check done. Metrics: %s", metrics)

    dataset_name = Path(dataset_path).name
    run_name = str(getattr(params, "run_name", "dl_sanity"))
    model_name = f"{run_name}_{dataset_name}"
    shaded_error_bar = build_shaded_error_bar_plot(x_small, y_small, model_name)

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
            "weight_classes": False,
            "weighted_sampler": False,
            "train_samples": int(len(x_small)),
            "val_samples": int(len(x_small)),
            "test_samples": int(len(x_small)),
            "run_name": run_name,
            "sanity_check": True,
            "sanity_subset_size": int(len(x_small)),
            "memorized": bool(metrics.get("accuracy", 0.0) >= 0.95),
        },
    }