"""ML testing action for evaluating saved models against selected datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ...models.dtos import BaseTaskDTO, MLTestDatasetModelParamsDTO
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


def _flatten_features(x: np.ndarray) -> np.ndarray:
    """Flatten input features to 2D for classical estimators."""
    if x.ndim <= 2:
        return x
    return x.reshape(x.shape[0], -1)


@register_ml("Test Feature Models", MLTestDatasetModelParamsDTO)
def test_feature_models(self, task_dto: BaseTaskDTO, params: MLTestDatasetModelParamsDTO):
    """Test selected or all saved models on selected or all discovered datasets."""
    task_name = getattr(task_dto, "task", None)
    all_models = self.discover_models(task_name) if hasattr(self, "discover_models") else []
    all_datasets = self.discover_datasets(task_name) if hasattr(self, "discover_datasets") else []

    selected_model_path = _selected_value(getattr(params, "model_path", None))
    selected_dataset_path = _selected_value(getattr(params, "dataset_path", None))

    if not all_models:
        return {"status": "no_models", "message": "No saved models discovered for testing."}
    if not all_datasets:
        return {"status": "no_datasets", "message": "No datasets discovered for testing."}

    if selected_model_path and selected_model_path != "__all__":
        models = [m for m in all_models if m["path"] == selected_model_path]
    else:
        models = all_models

    if selected_dataset_path and selected_dataset_path != "__all__":
        datasets = [d for d in all_datasets if d["path"] == selected_dataset_path]
    else:
        datasets = all_datasets

    rows: List[Dict[str, object]] = []
    for model_info in models:
        clf = load(model_info["path"])
        for dataset_info in datasets:
            arrays = _load_dataset(dataset_info["path"])
            x_test = _flatten_features(np.asarray(arrays["x"]))
            y_test = np.asarray(arrays["y"])
            group = np.asarray(arrays["group"])

            if x_test.shape[0] == 0:
                continue

            y_pred = clf.predict(x_test)

            rows.append(
                {
                    "model_name": model_info.get("name", Path(model_info["path"]).name),
                    "model_path": model_info["path"],
                    "estimator": model_info.get("estimator", "unknown"),
                    "dataset_name": dataset_info.get("name", Path(dataset_info["path"]).name),
                    "dataset_path": dataset_info["path"],
                    "n_test": int(len(y_test)),
                    "accuracy": float(accuracy_score(y_test, y_pred)),
                    "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
                    "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
                    "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
                }
            )

    df = pd.DataFrame(rows)
    return df
