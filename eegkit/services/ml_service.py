"""AI service: dataset building, simple training, and model registry introspection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from .ai_service import AIService

from .ml_actions import ml_registry  


class EEGMLService(AIService):
    """Provide Machine learning-related actions"""

    description = "Machine learning training and inference on epochs (registry-based)."

    def __init__(self, get_raw_func=None, get_epochs_func=None, get_task_func=None, get_subjects_metadata_func=None, jobs_root=None):
        """Initialize with controller callbacks and bind ML registry to spec."""
        super().__init__(
            registry=ml_registry,
            get_raw_func=get_raw_func,
            get_epochs_func=get_epochs_func,
            get_task_func=get_task_func,
            jobs_root=jobs_root,
        )
        self.get_subjects_metadata = get_subjects_metadata_func

    @staticmethod
    def _read_model_metadata(model_path: Path, default_run_name: str, default_estimator: str) -> tuple[str, str, str]:
        """Return (run_name, estimator, dataset_path) from sidecar json when available."""
        run_name = default_run_name
        estimator = default_estimator
        dataset_path = ""
        meta_path = model_path.with_suffix(".json")
        if not meta_path.exists():
            return run_name, estimator, dataset_path
        try:
            metadata = json.loads(meta_path.read_text())
            run_name = str(metadata.get("run_name", run_name))
            estimator = str(metadata.get("estimator", estimator))
            dataset_path = str(metadata.get("dataset_path", dataset_path))
        except Exception:
            pass
        return run_name, estimator, dataset_path


    def discover_models(self, task_name: Optional[str] = None) -> List[Dict[str, str]]:
        """Discover saved classical model artifacts under jobs task directories."""
        roots = [self.jobs_root / str(task_name)] if task_name else [self.jobs_root]

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
                    rel_parts = model_path.resolve().relative_to(self.jobs_root.resolve()).parts
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
                run_name, estimator, dataset_path = self._read_model_metadata(
                    model_path=model_path,
                    default_run_name=rel_parts[2],
                    default_estimator=estimator,
                )

                seen_paths.add(resolved)
                models.append(
                    {
                        "name": model_name,
                        "path": resolved,
                        "task": task_part,
                        "run_name": run_name,
                        "estimator": estimator,
                        "dataset_path": dataset_path,
                        "model_kind": "joblib",
                    }
                )

        models.sort(key=lambda item: (item.get("task", ""), item.get("run_name", ""), item["name"]))
        return models