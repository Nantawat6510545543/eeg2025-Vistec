"""ML training action driven by discovered saved feature datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from ...models.dtos import BaseTaskDTO, MLTrainDatasetParamsDTO
from ..ai_models.machine_learning import build_estimator
from . import register_ml


def _label_smoothing(y: np.ndarray, epsilon: float = 0.1, seed: int = 42) -> np.ndarray:
    """Randomly reassign ``epsilon`` fraction of training labels to another class.

    Acts as label-noise regularization for small or imbalanced datasets.
    Pass ``epsilon=0`` to disable (returns *y* unchanged).
    """
    if epsilon <= 0.0 or len(np.unique(y)) < 2:
        return y
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    y_smooth = y.copy()
    flip_idx = rng.choice(len(y), size=max(1, int(len(y) * epsilon)), replace=False)
    for i in flip_idx:
        other = classes[classes != y[i]]
        y_smooth[i] = rng.choice(other)
    return y_smooth


def _split_with_groups(
    x: np.ndarray,
    y: np.ndarray,
    group: np.ndarray,
    *,
    test_split: float,
    val_split: float,
    seed: int,
):
    """Split dataset into train/val/test while preserving group isolation when possible."""
    n_samples = x.shape[0]
    idx = np.arange(n_samples)

    if n_samples < 10 or len(np.unique(group.astype(str))) < 3:
        train_idx, test_idx = train_test_split(
            idx,
            test_size=test_split,
            random_state=seed,
            stratify=y if len(np.unique(y)) > 1 else None,
        )
        x_trainval, y_trainval = x[train_idx], y[train_idx]
        val_ratio = min(max(val_split / max(1e-9, 1.0 - test_split), 0.05), 0.5)
        tr_rel, val_rel = train_test_split(
            np.arange(len(train_idx)),
            test_size=val_ratio,
            random_state=seed,
            stratify=y_trainval if len(np.unique(y_trainval)) > 1 else None,
        )
        final_train_idx = train_idx[tr_rel]
        final_val_idx = train_idx[val_rel]
        return final_train_idx, final_val_idx, test_idx

    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_split, random_state=seed)
    trainval_rel, test_rel = next(gss_test.split(x, y, groups=group))
    trainval_idx = idx[trainval_rel]
    test_idx = idx[test_rel]

    val_ratio = min(max(val_split / max(1e-9, 1.0 - test_split), 0.05), 0.5)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    tr_rel, val_rel = next(gss_val.split(x[trainval_idx], y[trainval_idx], groups=group[trainval_idx]))
    final_train_idx = trainval_idx[tr_rel]
    final_val_idx = trainval_idx[val_rel]
    return final_train_idx, final_val_idx, test_idx


@register_ml("Train Feature Model", MLTrainDatasetParamsDTO)
def train_feature_model(self, task_dto: BaseTaskDTO, params: MLTrainDatasetParamsDTO):
    """Train a classical model from a discovered feature dataset."""
    selected_dataset = self.selected_value(getattr(params, "dataset_path", None))
    task_name = getattr(task_dto, "task", None)
    dataset_path = self.resolve_dataset_path(selected_dataset, task_name)

    if not dataset_path:
        return {
            "status": "no_dataset_selected",
            "message": "Select a discovered dataset before training.",
        }

    arrays = self.load_xyg_dataset(dataset_path)
    x = self.flatten_features_2d(np.asarray(arrays["x"]))
    y = np.asarray(arrays["y"])
    group = np.asarray(arrays["group"])

    if x.shape[0] == 0:
        return {
            "status": "empty_dataset",
            "message": "Selected dataset has no rows.",
        }

    estimator_name = self.selected_value(getattr(params, "estimator", None)) or "svm"
    train_idx, val_idx, test_idx = _split_with_groups(
        x,
        y,
        group,
        test_split=float(getattr(params, "test_split", 0.2)),
        val_split=float(getattr(params, "val_split", 0.2)),
        seed=int(getattr(params, "seed", 42)),
    )

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    label_smoothing = float(getattr(params, "label_smoothing", 0.0))
    y_train = _label_smoothing(y_train, epsilon=label_smoothing, seed=int(getattr(params, "seed", 42)))

    run_name = getattr(params, "run_name", "ml_train")
    dataset_name = Path(dataset_path).stem if Path(dataset_path).is_file() else Path(dataset_path).name
    model_name = f"{run_name}_{dataset_name}_{estimator_name}"
    estimator = build_estimator(
        estimator_name,
        model_name=model_name,
        random_state=int(getattr(params, "seed", 42)),
    )
    fit_info = estimator.fit(x_train, y_train, x_val, y_val)
    trained_model = fit_info.get("model") if isinstance(fit_info, dict) else None
    _, metrics = estimator.predict(x_test, y_test, classifier=trained_model)

    model_info = {k: v for k, v in fit_info.items() if k != "model"}

    return {
        "model": trained_model,
        "model_name": model_name,
        "model_info": model_info,
        "evaluation": metrics,
    }