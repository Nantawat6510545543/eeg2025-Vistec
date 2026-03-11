"""AI service: dataset building, simple training, and model registry introspection."""

from __future__ import annotations

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

    def prepare_params(self, task_dto, params_dto):
        """Return dynamic parameter choices for ML actions including discovered datasets."""
        updates = super().prepare_params(task_dto, params_dto)
        if hasattr(params_dto, "dataset_path"):
            datasets = self.discover_datasets(getattr(task_dto, "task", None))
            updates["dataset_path"] = [
                (
                    f"{item['name']} [{item['kind']}] - {item.get('action', 'unknown')}/{item.get('session', 'unknown')}",
                    item["path"],
                )
                for item in datasets
            ] or [None]
        return updates
