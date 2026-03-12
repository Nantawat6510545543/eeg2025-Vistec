"""AI service: dataset building, simple training, and model registry introspection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .base_service import BaseService

from .ml_actions import ml_registry  


class EEGMLService(BaseService):
    """Provide Machine learning-related actions"""

    description = "Machine learning training and inference on epochs (registry-based)."

    def __init__(self, get_raw_func=None, get_epochs_func=None, get_task_func=None, get_subjects_metadata_func=None, jobs_root=None):
        """Initialize with controller callbacks and bind ML registry to spec."""
        super().__init__(
            registry=ml_registry,
            get_raw_func=get_raw_func,
            get_epochs_func=get_epochs_func,
            get_task_func=get_task_func,
        )
        self.get_subjects_metadata = get_subjects_metadata_func
        self.jobs_root = Path(jobs_root or Path.cwd() / "jobs")

    def discover_datasets(self, task_name: str | None = None) -> List[Dict[str, str]]:
        """Discover saved datasets under the known jobs layout.

        Expected directory form:
        jobs/<task>/<action>/<session>/<dataset_name>/{x.npy,y.npy,group.npy}
        """
        jobs_root = self.jobs_root
        if task_name:
            roots = [jobs_root / str(task_name)]
        else:
            roots = [jobs_root]

        datasets: List[Dict[str, str]] = []
        seen_paths: set[str] = set()

        for root in roots:
            if not root.exists():
                continue

            for x_path in root.rglob("x.npy"):
                dataset_dir = x_path.parent
                y_path = dataset_dir / "y.npy"
                group_path = dataset_dir / "group.npy"
                if not (y_path.exists() and group_path.exists()):
                    continue

                # Require the saved dataset to live under the expected jobs hierarchy.
                try:
                    rel_parts = dataset_dir.resolve().relative_to(jobs_root.resolve()).parts
                except Exception:
                    continue
                if len(rel_parts) < 4:
                    continue

                task_part, action_part, session_part = rel_parts[0], rel_parts[1], rel_parts[2]
                resolved = str(dataset_dir.resolve())
                if resolved in seen_paths:
                    continue
                seen_paths.add(resolved)
                datasets.append({
                    "name": dataset_dir.name,
                    "path": resolved,
                    "kind": "directory",
                    "task": task_part,
                    "action": action_part,
                    "session": session_part,
                })

            for npz_path in root.rglob("*.npz"):
                resolved = str(npz_path.resolve())
                if resolved in seen_paths:
                    continue
                try:
                    rel_parts = npz_path.resolve().relative_to(jobs_root.resolve()).parts
                except Exception:
                    continue
                if len(rel_parts) < 4:
                    continue
                try:
                    with np.load(npz_path, allow_pickle=False) as archive:
                        if not {"x", "y", "group"}.issubset(set(archive.files)):
                            continue
                except Exception:
                    continue
                task_part, action_part, session_part = rel_parts[0], rel_parts[1], rel_parts[2]
                seen_paths.add(resolved)
                datasets.append({
                    "name": npz_path.stem,
                    "path": resolved,
                    "kind": "npz",
                    "task": task_part,
                    "action": action_part,
                    "session": session_part,
                })

        datasets.sort(key=lambda item: (item.get("task", ""), item["name"], item["path"]))
        return datasets

    def discover_models(self, task_name: str | None = None) -> List[Dict[str, str]]:
        """Discover saved classical model artifacts under jobs task directories."""
        jobs_root = self.jobs_root
        if task_name:
            roots = [jobs_root / str(task_name)]
        else:
            roots = [jobs_root]

        models: List[Dict[str, str]] = []
        seen_paths: set[str] = set()

        for root in roots:
            if not root.exists():
                continue

            for model_path in root.rglob("*_model.gz"):
                resolved = str(model_path.resolve())
                if resolved in seen_paths:
                    continue

                try:
                    rel_parts = model_path.resolve().relative_to(jobs_root.resolve()).parts
                except Exception:
                    continue
                if len(rel_parts) < 4:
                    continue

                task_part = rel_parts[0]
                model_name = model_path.stem.replace("_model", "")
                estimator = "unknown"
                for candidate in ("random_forest", "svm", "knn"):
                    if model_name.endswith(candidate):
                        estimator = candidate
                        break

                meta_path = model_path.with_suffix(".json")
                dataset_path = ""
                run_name = rel_parts[2]
                if meta_path.exists():
                    try:
                        metadata = json.loads(meta_path.read_text())
                        estimator = str(metadata.get("estimator", estimator))
                        dataset_path = str(metadata.get("dataset_path", ""))
                        run_name = str(metadata.get("run_name", run_name))
                    except Exception:
                        pass

                seen_paths.add(resolved)
                models.append({
                    "name": model_name,
                    "path": resolved,
                    "task": task_part,
                    "run_name": run_name,
                    "estimator": estimator,
                    "dataset_path": dataset_path,
                })

        models.sort(key=lambda item: (item.get("task", ""), item.get("run_name", ""), item["name"]))
        return models

    @staticmethod
    def dataset_option_key(item: Dict[str, str]) -> str:
        """Return stable dataset dropdown key in datasetname-jobid format."""
        return f"{item.get('name', 'dataset')}-{item.get('session', 'unknown')}"

    @staticmethod
    def model_option_key(item: Dict[str, str]) -> str:
        """Return stable model dropdown key in modelname-jobid format."""
        return f"{item.get('name', 'model')}-{item.get('run_name', 'run')}"

    def resolve_dataset_path(self, selected: str | None, task_name: str | None = None) -> str | None:
        """Resolve selected dataset key/path to a real dataset path."""
        if not selected or selected == "__all__":
            return selected
        p = Path(str(selected))
        if p.exists():
            return str(p.resolve())
        for item in self.discover_datasets(task_name):
            if selected in {item.get("path"), self.dataset_option_key(item), item.get("name")}:
                return item.get("path")
        return None

    def resolve_model_path(self, selected: str | None, task_name: str | None = None) -> str | None:
        """Resolve selected model key/path to a real model artifact path."""
        if not selected or selected == "__all__":
            return selected
        p = Path(str(selected))
        if p.exists():
            return str(p.resolve())
        for item in self.discover_models(task_name):
            if selected in {item.get("path"), self.model_option_key(item), item.get("name")}:
                return item.get("path")
        return None

    def prepare_params(self, task_dto, params_dto):
        """Return dynamic parameter choices for ML actions including discovered datasets."""
        updates = super().prepare_params(task_dto, params_dto)
        task_name = getattr(task_dto, "task", None)
        if hasattr(params_dto, "dataset_path"):
            datasets = self.discover_datasets(task_name)
            updates["dataset_path"] = [self.dataset_option_key(item) for item in datasets] or [None]

        if hasattr(params_dto, "model_path"):
            models = self.discover_models(task_name)
            updates["model_path"] = [self.model_option_key(item) for item in models]
            if updates["model_path"]:
                updates["model_path"] = updates["model_path"]
            else:
                updates["model_path"] = [None]
        return updates
