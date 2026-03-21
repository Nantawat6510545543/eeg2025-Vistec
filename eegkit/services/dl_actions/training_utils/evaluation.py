"""Evaluation loops and metrics for binary DL classifiers."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader


def eval_loop(
    model: nn.Module,
    te_dl: DataLoader,
    device: torch.device,
    classification: bool = True,
):
    """Return arrays from test loader for classification or regression."""
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for x_b, y_b in te_dl:
            x_b = x_b.to(device)
            outputs = model(x_b)
            if classification:
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                preds = outputs.argmax(dim=1).cpu().numpy()
                y_true.extend(y_b.numpy())
                y_pred.extend(preds)
                y_prob.extend(probs)
            else:
                preds = outputs.detach().cpu().numpy().reshape(-1)
                y_pred.extend(preds.tolist())
                y_true.extend(y_b.numpy().reshape(-1).tolist())

    if classification:
        return np.array(y_true), np.array(y_pred), np.array(y_prob)
    return np.asarray(y_true, dtype=np.float32), np.asarray(y_pred, dtype=np.float32)


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict:
    """Return classification metrics including imbalance-robust diagnostics."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    auc: Optional[float] = None
    pr_auc: Optional[float] = None
    if len(set(y_true.tolist())) == 2:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            pass
        try:
            pr_auc = float(average_precision_score(y_true, y_prob))
        except Exception:
            pass

    class_recalls = recall_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)
    class_precisions = precision_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)

    prob_pos_true_0 = y_prob[y_true == 0]
    prob_pos_true_1 = y_prob[y_true == 1]

    stats_0_mean = float(np.mean(prob_pos_true_0)) if prob_pos_true_0.size > 0 else 0.0
    stats_0_std = float(np.std(prob_pos_true_0)) if prob_pos_true_0.size > 0 else 0.0
    stats_1_mean = float(np.mean(prob_pos_true_1)) if prob_pos_true_1.size > 0 else 0.0
    stats_1_std = float(np.std(prob_pos_true_1)) if prob_pos_true_1.size > 0 else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_binary": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "roc_auc": auc,
        "pr_auc": pr_auc,
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": cm.astype(int).tolist(),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "class_recall": {
            "0": float(class_recalls[0]),
            "1": float(class_recalls[1]),
        },
        "class_precision": {
            "0": float(class_precisions[0]),
            "1": float(class_precisions[1]),
        },
        "class_prob_pos_stats": {
            "0": {
                "mean": stats_0_mean,
                "std": stats_0_std,
            },
            "1": {
                "mean": stats_1_mean,
                "std": stats_1_std,
            },
        },
    }


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Return common single-target regression metrics."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    evs = float(explained_variance_score(y_true, y_pred))
    y_true_mean = float(np.mean(y_true))
    y_pred_mean = float(np.mean(y_pred))
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else float("nan")

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "explained_variance": evs,
        "pearson_corr": corr,
        "y_true_mean": y_true_mean,
        "y_pred_mean": y_pred_mean,
    }
