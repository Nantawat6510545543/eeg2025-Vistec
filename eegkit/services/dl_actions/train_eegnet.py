"""Train a binary EEGNet on a saved DL epoch tensor dataset."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset as TorchDataset, WeightedRandomSampler
from tqdm.auto import tqdm

from ...models.dtos import BaseTaskDTO, DLTrainParamsDTO
from ...services.ai_models.deep_learning.EEGNetBinary import EEGNetBinary
from . import register_dl

logger = logging.getLogger(__name__)
class _NumpyDataset(TorchDataset):
    """Wrap numpy arrays as a minimal torch Dataset."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.as_tensor(self.x[idx], dtype=torch.float32),
            torch.as_tensor(self.y[idx], dtype=torch.long),
        )


def _split_with_groups(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    test_split: float,
    val_split: float,
    seed: int,
):
    """Subject-independent train / val / test split with configurable ratios.

    ``test_split`` and ``val_split`` are ratios of total samples.
    The effective validation ratio on the post-test remainder is computed as:
    ``val_split / (1 - test_split)``.
    """
    if test_split <= 0.0 or test_split >= 1.0:
        raise ValueError(f"test_split must be in (0, 1), got {test_split}")
    if val_split <= 0.0 or val_split >= 1.0:
        raise ValueError(f"val_split must be in (0, 1), got {val_split}")
    if test_split + val_split >= 1.0:
        raise ValueError(
            f"test_split + val_split must be < 1.0, got {test_split + val_split}"
        )

    # GroupShuffleSplit.split() returns (train_idx, test_idx).
    # Use gss1 to hold out the test set, then split the remainder into train/val.
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_split, random_state=seed)
    trainval_idx, test_idx = next(gss1.split(x, y, groups=groups))

    x_tv, y_tv, g_tv = x[trainval_idx], y[trainval_idx], groups[trainval_idx]
    val_ratio = val_split / (1.0 - test_split)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed + 1)
    train_idx, val_idx = next(gss2.split(x_tv, y_tv, groups=g_tv))

    return (
        x_tv[train_idx], y_tv[train_idx],
        x_tv[val_idx], y_tv[val_idx],
        x[test_idx], y[test_idx],
    )


def _build_dataloaders(
    x_tr, y_tr, x_val, y_val, x_te, y_te, batch_size: int, weighted_sampler: bool
):
    """Return train / val / test DataLoaders."""
    sampler: Optional[WeightedRandomSampler] = None
    shuffle = True
    if weighted_sampler:
        counts = np.bincount(y_tr, minlength=2)
        counts = np.maximum(counts, 1)
        w = (1.0 / counts)[y_tr]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(w, dtype=torch.double),
            num_samples=len(w),
            replacement=True,
        )
        shuffle = False

    tr_dl = DataLoader(
        _NumpyDataset(x_tr, y_tr),
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=False,
    )
    val_dl = DataLoader(_NumpyDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    te_dl = DataLoader(_NumpyDataset(x_te, y_te), batch_size=batch_size, shuffle=False)
    return tr_dl, val_dl, te_dl


def _train_loop(
    model: nn.Module,
    tr_dl: DataLoader,
    val_dl: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epochs: int,
    device: torch.device,
    es_patience: int,
) -> nn.Module:
    """Run training with optional early stopping; return model with best val loss."""
    best_val = float("inf")
    no_improve = 0
    best_state: Optional[Dict] = None

    pbar = tqdm(range(epochs), desc="Training", unit="epoch", dynamic_ncols=True)
    for epoch in pbar:
        model.train()
        for x_b, y_b in tr_dl:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss_fn(model(x_b), y_b).backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_b, y_b in val_dl:
                x_b, y_b = x_b.to(device), y_b.to(device)
                val_loss += loss_fn(model(x_b), y_b).item()
        val_loss /= max(len(val_dl), 1)
        scheduler.step(val_loss)

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        pbar.set_postfix(val_loss=f"{val_loss:.4f}", best=f"{best_val:.4f}", improved=improved)
        logger.debug("epoch %d/%d  val_loss=%.4f", epoch + 1, epochs, val_loss)

        if es_patience > 0 and no_improve >= es_patience:
            logger.info("early stopping triggered at epoch %d", epoch + 1)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _eval_loop(model: nn.Module, te_dl: DataLoader, device: torch.device):
    """Return (y_true, y_pred, y_prob_pos) arrays from test loader."""
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for x_b, y_b in te_dl:
            x_b = x_b.to(device)
            logits = model(x_b)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            y_true.extend(y_b.numpy())
            y_pred.extend(preds)
            y_prob.extend(probs)
    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict:
    """Return a dict of classification metrics."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    auc: Optional[float] = None
    if len(set(y_true.tolist())) == 2:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            pass

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_binary": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "roc_auc": auc,
    }


class _FocalLoss(nn.Module):
    """Multi-class focal loss for logits."""

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


def _class_weights(y, device: torch.device) -> Optional[torch.Tensor]:
    """Compute balanced class weights from labels and move to target device."""
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    labels = np.asarray(y).astype(int)
    classes = np.unique(labels)
    if classes.size <= 1:
        return None
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return torch.from_numpy(np.asarray(cw)).float().to(device)


def _build_loss_function(
    name: str,
    class_weights: Optional[torch.Tensor],
    label_smoothing: float,
) -> nn.Module:
    """Create configured loss module from dropdown selection."""
    if name == "focal_loss":
        return _FocalLoss(gamma=2.0, weight=class_weights)
    return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)


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

    test_split = float(getattr(params, "test_split", 0.25))
    val_split = float(getattr(params, "val_split", 0.25))
    x_tr, y_tr, x_val, y_val, x_te, y_te = _split_with_groups(
        x,
        y,
        groups,
        test_split=test_split,
        val_split=val_split,
        seed=int(getattr(params, "seed", 42)),
    )

    device_str = self.selected_value(getattr(params, "device", "cpu")) or "cpu"
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    elif device_str.startswith("cuda") and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    seed = int(getattr(params, "seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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

    weight_tensor = _class_weights(y_tr, device) if use_class_weights else None

    loss_choice = self.selected_value(getattr(params, "loss_function", "cross_entropy"))
    loss_name = str(loss_choice or "cross_entropy")
    label_smoothing = float(getattr(params, "label_smoothing", 0.0))
    loss_fn = _build_loss_function(loss_name, weight_tensor, label_smoothing)
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

    tr_dl, val_dl, te_dl = _build_dataloaders(
        x_tr, y_tr, x_val, y_val, x_te, y_te, batch_size, weighted_sampler
    )
    logger.info(
        "Training EEGNet: n_ch=%d n_t=%d device=%s train/val/test=%d/%d/%d",
        n_channels, n_timepoints, device_str, len(x_tr), len(x_val), len(x_te),
    )

    model = _train_loop(
        model, tr_dl, val_dl, loss_fn, optimizer, scheduler, epochs_n, device, es_patience
    )
    y_true, y_pred, y_prob = _eval_loop(model, te_dl, device)
    metrics = _compute_metrics(y_true, y_pred, y_prob)
    logger.info("EEGNet training done. Metrics: %s", metrics)

    dataset_name = Path(dataset_path).name
    run_name = str(getattr(params, "run_name", "dl_train"))
    model_name = f"{run_name}_{dataset_name}"

    return {
        "model": model,
        "model_name": model_name,
        "evaluation": metrics,
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
