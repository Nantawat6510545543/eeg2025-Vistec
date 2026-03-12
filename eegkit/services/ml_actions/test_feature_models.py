"""ML testing action for evaluating saved models against selected datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from joblib import load

from ...models.dtos import BaseTaskDTO, MLTestDatasetModelParamsDTO
from . import register_ml


def _build_true_vs_pred_figure(pred_records: List[Dict[str, object]]):
    """Create a per-run true-vs-pred scatter plot figure."""
    n_panels = max(1, len(pred_records))
    fig, axes = plt.subplots(n_panels, 1, figsize=(8, max(4, 3 * n_panels)), squeeze=False)
    axes = axes.ravel()

    for ax, rec in zip(axes, pred_records):
        y_true = np.asarray(rec["y_true"])  # type: ignore[index]
        y_pred = np.asarray(rec["y_pred"])  # type: ignore[index]

        classes = np.unique(np.concatenate([y_true, y_pred]))
        class_to_idx = {c: i for i, c in enumerate(classes)}
        true_idx = np.array([class_to_idx[c] for c in y_true], dtype=float)
        pred_idx = np.array([class_to_idx[c] for c in y_pred], dtype=float)

        jitter = 0.08
        rng = np.random.default_rng(42)
        xj = true_idx + rng.normal(0.0, jitter, size=true_idx.shape[0])
        yj = pred_idx + rng.normal(0.0, jitter, size=pred_idx.shape[0])

        ax.scatter(xj, yj, s=18, alpha=0.55)
        lim = (-0.5, len(classes) - 0.5)
        ax.plot(lim, lim, "k--", linewidth=1.0)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))
        ax.set_xticklabels([str(c) for c in classes], rotation=45, ha="right")
        ax.set_yticklabels([str(c) for c in classes])
        ax.set_xlabel("True")
        ax.set_ylabel("Pred")
        ax.set_title(f"{rec['model_name']} | {rec['dataset_name']}")
        ax.grid(alpha=0.25)

    for ax in axes[len(pred_records):]:
        ax.axis("off")

    fig.tight_layout()
    return fig


@register_ml("Test Feature Models", MLTestDatasetModelParamsDTO)
def test_feature_models(self, task_dto: BaseTaskDTO, params: MLTestDatasetModelParamsDTO):
    """Test selected model/dataset and return a true-vs-pred plot."""
    task_name = getattr(task_dto, "task", None)
    all_models = self.discover_models(task_name)
    all_datasets = self.discover_datasets(task_name)

    selected_model = self.selected_value(getattr(params, "model_path", None))
    selected_dataset = self.selected_value(getattr(params, "dataset_path", None))
    selected_model_path = self.resolve_model_path(selected_model, task_name)
    selected_dataset_path = self.resolve_dataset_path(selected_dataset, task_name)

    if not all_models:
        return {"status": "no_models", "message": "No saved models discovered for testing."}
    if not all_datasets:
        return {"status": "no_datasets", "message": "No datasets discovered for testing."}

    if selected_model_path:
        models = [m for m in all_models if m["path"] == selected_model_path]
    else:
        models = all_models

    if selected_dataset_path:
        datasets = [d for d in all_datasets if d["path"] == selected_dataset_path]
    else:
        datasets = all_datasets

    pred_records: List[Dict[str, object]] = []
    for model_info in models:
        clf = load(model_info["path"])
        for dataset_info in datasets:
            arrays = self.load_xyg_dataset(dataset_info["path"])
            x_test = self.flatten_features_2d(np.asarray(arrays["x"]))
            y_test = np.asarray(arrays["y"])

            if x_test.shape[0] == 0:
                continue

            y_pred = clf.predict(x_test)
            pred_records.append(
                {
                    "model_name": model_info.get("name", Path(model_info["path"]).name),
                    "dataset_name": dataset_info.get("name", Path(dataset_info["path"]).name),
                    "y_true": y_test,
                    "y_pred": y_pred,
                }
            )
    return _build_true_vs_pred_figure(pred_records)
