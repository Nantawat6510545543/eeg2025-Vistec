"""Evaluation loops and metrics for n-class DL classifiers."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader


def eval_loop_multiclass(model: nn.Module, te_dl: DataLoader, device: torch.device):
    """Return (y_true, y_pred, y_prob_all) for n-class classification."""
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for x_b, y_b in te_dl:
            x_b = x_b.to(device)
            logits = model(x_b)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            y_true.extend(y_b.numpy())
            y_pred.extend(preds)
            y_prob.append(probs)

    y_prob_all = np.concatenate(y_prob, axis=0) if y_prob else np.empty((0, 0), dtype=np.float32)
    return np.array(y_true), np.array(y_pred), y_prob_all


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob_all: np.ndarray,
    labels: Optional[np.ndarray] = None,
) -> Dict:
    """Return robust n-class metrics with confusion matrix and per-class stats."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = np.asarray(labels).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    class_recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    class_precisions = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    roc_auc_ovr_macro: Optional[float] = None
    if y_prob_all.ndim == 2 and y_prob_all.shape[0] == y_true.shape[0] and y_prob_all.shape[1] >= len(labels):
        try:
            y_true_bin = label_binarize(y_true, classes=labels)
            y_score = y_prob_all[:, : len(labels)]
            roc_auc_ovr_macro = float(
                roc_auc_score(y_true_bin, y_score, average="macro", multi_class="ovr")
            )
        except Exception:
            pass

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "roc_auc_ovr_macro": roc_auc_ovr_macro,
        "confusion_matrix": {
            "labels": labels.astype(int).tolist(),
            "matrix": cm.astype(int).tolist(),
        },
        "class_recall": {
            str(int(lbl)): float(class_recalls[i]) for i, lbl in enumerate(labels)
        },
        "class_precision": {
            str(int(lbl)): float(class_precisions[i]) for i, lbl in enumerate(labels)
        },
        "class_support": {
            str(int(lbl)): int((y_true == lbl).sum()) for lbl in labels
        },
    }
    return metrics
