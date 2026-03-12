"""AI service superclass shared by ML and DL services."""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .base_service import BaseService


# ---------------------------------------------------------------------------

class AIService(BaseService):
    """Base class for registry-driven AI services that consume saved datasets."""

    def __init__(
        self,
        *,
        registry,
        get_raw_func=None,
        get_epochs_func=None,
        get_evoked_func=None,
        get_task_func=None,
        jobs_root=None,
    ):
        super().__init__(
            registry=registry,
            get_raw_func=get_raw_func,
            get_epochs_func=get_epochs_func,
            get_evoked_func=get_evoked_func,
            get_task_func=get_task_func,
        )
        self.jobs_root = Path(jobs_root or Path.cwd() / "jobs")

    # ------------------------------------------------------------------
    # Static helpers — callable as self.method() from action functions
    # ------------------------------------------------------------------

    @staticmethod
    def selected_value(value):
        """Normalize dropdown-style list values to a single selection."""
        if isinstance(value, (list, tuple)):
            return value[0] if value else None
        return value

    @staticmethod
    def load_xyg_dataset(dataset_path: str) -> Dict[str, np.ndarray]:
        """Load x/y/group arrays from a directory of NPY files or an NPZ archive."""
        path = Path(dataset_path)
        if path.is_dir():
            return {
                "x": np.load(path / "x.npy", allow_pickle=False),
                "y": np.load(path / "y.npy", allow_pickle=False),
                "group": np.load(path / "group.npy", allow_pickle=True),
            }
        if path.is_file() and path.suffix == ".npz":
            with np.load(path, allow_pickle=True) as archive:
                return {"x": archive["x"], "y": archive["y"], "group": archive["group"]}
        raise FileNotFoundError(f"Unsupported dataset path: {dataset_path}")

    @staticmethod
    def flatten_features_2d(x: np.ndarray) -> np.ndarray:
        """Flatten input features to 2D for classical estimators."""
        if x.ndim <= 2:
            return x
        return x.reshape(x.shape[0], -1)

    def discover_datasets(self, task_name: Optional[str] = None) -> List[Dict[str, str]]:
        """Discover saved datasets under jobs/<task>/<action>/<session>/<dataset>."""
        roots = [self.jobs_root / str(task_name)] if task_name else [self.jobs_root]
        datasets: List[Dict[str, str]] = []
        seen: set[str] = set()

        for root in roots:
            if not root.exists():
                continue

            for x_path in root.rglob("x.npy"):
                ds_dir = x_path.parent
                if not ((ds_dir / "y.npy").exists() and (ds_dir / "group.npy").exists()):
                    continue
                try:
                    parts = ds_dir.resolve().relative_to(self.jobs_root.resolve()).parts
                except Exception:
                    continue
                if len(parts) < 4:
                    continue
                resolved = str(ds_dir.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)
                datasets.append({
                    "name": ds_dir.name, "path": resolved, "kind": "directory",
                    "task": parts[0], "action": parts[1], "session": parts[2],
                })

            for npz_path in root.rglob("*.npz"):
                resolved = str(npz_path.resolve())
                if resolved in seen:
                    continue
                try:
                    parts = npz_path.resolve().relative_to(self.jobs_root.resolve()).parts
                except Exception:
                    continue
                if len(parts) < 4:
                    continue
                try:
                    with np.load(npz_path, allow_pickle=False) as arc:
                        if not {"x", "y", "group"}.issubset(set(arc.files)):
                            continue
                except Exception:
                    continue
                seen.add(resolved)
                datasets.append({
                    "name": npz_path.stem, "path": resolved, "kind": "npz",
                    "task": parts[0], "action": parts[1], "session": parts[2],
                })

        datasets.sort(key=lambda d: (d.get("task", ""), d["name"], d["path"]))
        return datasets

    @staticmethod
    def dataset_option_key(item: Dict[str, str]) -> str:
        """Return stable dataset dropdown key in datasetname-sessionid format."""
        return f"{item.get('name', 'dataset')}-{item.get('session', 'unknown')}"

    def resolve_dataset_path(self, selected: Optional[str], task_name: Optional[str] = None) -> Optional[str]:
        """Resolve selected key/path to an actual dataset path."""
        if not selected or selected == "__all__":
            return selected
        p = Path(str(selected))
        if p.exists():
            return str(p.resolve())
        for item in self.discover_datasets(task_name):
            if selected in {item.get("path"), self.dataset_option_key(item), item.get("name")}:
                return item.get("path")
        return None

    @abstractmethod
    def discover_models(self, task_name: Optional[str] = None) -> List[Dict[str, str]]:
        """Discover saved model artifacts under jobs task directories."""
        raise NotImplementedError

    @staticmethod
    def model_option_key(item: Dict[str, str]) -> str:
        """Return stable model dropdown key in modelname-runid format."""
        return f"{item.get('name', 'model')}-{item.get('run_name', 'run')}"

    def resolve_model_path(self, selected: Optional[str], task_name: Optional[str] = None) -> Optional[str]:
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
        """Add dynamic dataset/model choices on top of BaseService.prepare_params."""
        updates = super().prepare_params(task_dto, params_dto)
        task_name = getattr(task_dto, "task", None)
        if hasattr(params_dto, "dataset_path"):
            datasets = self.discover_datasets(task_name)
            updates["dataset_path"] = [self.dataset_option_key(d) for d in datasets] or [None]
        if hasattr(params_dto, "model_path"):
            models = self.discover_models(task_name)
            updates["model_path"] = [self.model_option_key(m) for m in models] or [None]
        return updates
