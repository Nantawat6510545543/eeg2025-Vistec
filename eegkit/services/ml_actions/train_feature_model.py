"""Placeholder ML training action driven by discovered saved datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from ...models.dtos import BaseTaskDTO, MLTrainDatasetParamsDTO
from . import register_ml


def _selected_value(value):
    """Normalize dropdown-style list values to a single selection."""
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def _load_dataset(dataset_path: str) -> Dict[str, np.ndarray]:
    """Load dataset arrays from a directory of NPY files or an NPZ archive."""
    path = Path(dataset_path)
    if path.is_dir():
        return {
            "x": np.load(path / "x.npy", allow_pickle=False),
            "y": np.load(path / "y.npy", allow_pickle=False),
            "group": np.load(path / "group.npy", allow_pickle=True),
        }

    if path.is_file() and path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as archive:
            return {
                "x": archive["x"],
                "y": archive["y"],
                "group": archive["group"],
            }

    raise FileNotFoundError(f"Unsupported dataset path: {dataset_path}")


@register_ml("Train Feature Model", MLTrainDatasetParamsDTO)
def train_feature_model(self, task_dto: BaseTaskDTO, params: MLTrainDatasetParamsDTO):
    """Load a discovered dataset and return a training placeholder summary."""
    dataset_path = _selected_value(getattr(params, "dataset_path", None))
    discovered = self.discover_datasets() if hasattr(self, "discover_datasets") else []

    if not dataset_path:
        return {
            "status": "no_dataset_selected",
            "message": "Select a discovered dataset before training.",
            "available_datasets": discovered,
        }

    arrays = _load_dataset(dataset_path)
    x = arrays["x"]
    y = arrays["y"]
    group = arrays["group"]

    unique_labels, label_counts = np.unique(y, return_counts=True) if len(y) > 0 else (np.array([]), np.array([]))
    unique_groups = np.unique(group.astype(str)) if len(group) > 0 else np.array([])

    return {
        "status": "placeholder",
        "message": "Dataset loading and selection are implemented. Model training is not implemented yet.",
        "run_name": params.run_name,
        "estimator": _selected_value(getattr(params, "estimator", None)),
        "dataset_path": str(Path(dataset_path)),
        "dataset_name": Path(dataset_path).stem if Path(dataset_path).is_file() else Path(dataset_path).name,
        "x_shape": list(x.shape),
        "y_shape": list(y.shape),
        "group_shape": list(group.shape),
        "n_groups": int(len(unique_groups)),
        "class_balance": {str(label): int(count) for label, count in zip(unique_labels.tolist(), label_counts.tolist())},
        "available_datasets": discovered,
    }