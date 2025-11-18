"""AI service: dataset building, simple training, and model registry introspection."""

from __future__ import annotations

from importlib import import_module
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .base_service import BaseService
from ..models.dtos import (
    BaseTaskDTO,
    EpochParamsDTO,
    FilterParamsDTO,
    AIBaseDTO,
    AITrainParamsDTO,
    AIPredictParamsDTO,
)

ai_registry: Dict[str, Dict[str, Any]] = {}


def register_ai(name: str, dto_cls):
    """Register an AI action with its params DTO class and handler function."""

    def _decorator(func):
        ai_registry[name] = {"params": dto_cls, "function": func}
        return func

    return _decorator


class EEGAIService(BaseService):
    """Provide AI-related actions (list models, build dataset, train, predict placeholder)."""

    description = "AI training and inference on epochs (registry-based)."

    def __init__(self, *, get_raw_func=None, get_epochs_func=None, get_task_func=None):
        """Initialize with controller callbacks and bind AI registry to spec."""
        super().__init__(
            registry=ai_registry,
            get_raw_func=get_raw_func,
            get_epochs_func=get_epochs_func,
            get_task_func=get_task_func,
        )

    # ---- helpers ----
    def _model_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Return mapping of model_name -> {display_name, description} discovered under ai_models."""
        meta: Dict[str, Dict[str, Any]] = {}
        try:
            mod = import_module('eegkit.services.ai_models')
            for attr in getattr(mod, '__all__', []):
                obj = getattr(mod, attr, None)
                if nn and isinstance(obj, type) and issubclass(obj, nn.Module):
                    display = getattr(obj, 'DISPLAY_NAME', attr)
                    desc = getattr(obj, 'DESCRIPTION', (obj.__doc__ or '').strip())
                    meta[attr] = {
                        'display_name': display,
                        'description': desc,
                    }
        except Exception:
            pass
        if not meta:
            meta['none'] = {
                'display_name': 'No models found',
                'description': 'No models discovered under eegkit.services.ai_models. '
                               'Add model classes and export them via __all__.'
            }
        return meta

    def _available_models(self) -> List[str]:
        return list(self._model_metadata().keys())

    def _build_epoch_dataset(self, task_dto: BaseTaskDTO, params: EpochParamsDTO):
        """Return (X, y, meta) with epochs transformed into numpy arrays.

        Detached helper (no mediator class) per user request.
        """
        epochs, labels = self.get_epochs(task_dto, params)
        if epochs is None:
            return None, None, {"reason": "epochs_unavailable"}

        epochs.load_data()

        X = epochs.get_data()  # (n_epochs, n_channels, n_times)
        if labels is None:
            inv_map = {v: k for k, v in (epochs.event_id or {}).items()}
            y = np.array([inv_map.get(code, "?") for code in epochs.events[:, 2]], dtype=object)
        else:
            y = np.array(labels)
        meta = {
            "sfreq": float(epochs.info.get("sfreq", 0.0)),
            "ch_names": list(epochs.ch_names),
            "event_id": dict(epochs.event_id or {}),
            "shape": tuple(X.shape),
        }
        return X.astype(np.float32), y, meta

    # ---- training/eval helpers ----
    def _select_device(self, pref: str | None = None):
        if torch is None:
            return None
        if pref == "cpu":
            return torch.device("cpu")
        if pref == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _encode_labels(self, y: np.ndarray):
        # Map unique labels -> indices
        uniq = np.unique(y)
        index = {v: i for i, v in enumerate(uniq)}
        y_idx = np.array([index[v] for v in y], dtype=np.int64)
        classes = [str(v) for v in uniq.tolist()]
        return y_idx, classes

    def _model_factory(self, name: str, n_channels: int, n_times: int, n_classes: int):
        """Instantiate supported models by internal name.

        Supported now:
        - SimpleNN: flatten input (C*T)
        - CNNLSTMDense: expects [B, C, T]
        Other models can be added later with proper input adapters.
        """
        mod = import_module('eegkit.services.ai_models')
        cls = getattr(mod, name, None)
        if cls is None:
            return None, {"reason": f"unknown model '{name}'"}
        try:
            if name == "SimpleNN":
                model = cls(input_dim=int(n_channels * n_times), num_classes=int(n_classes))
                return model, {"input_adapter": "flatten"}
            if name == "CNNLSTMDense":
                model = cls(in_channels=int(n_channels), num_classes=int(n_classes))
                return model, {"input_adapter": "channels_times"}
            return None, {"reason": f"model '{name}' not yet supported by trainer"}
        except Exception as e:
            return None, {"reason": f"model init error: {e}"}

    def prepare_params(self, task_dto, params_dto):
        """Extend BaseService.prepare_params and inject AI model choices.

        - Retains epoch label enrichment from the superclass
        - If params belong to AIBase, add {"model": [list of model names]} for UI dropdowns
        """
        updates = super().prepare_params(task_dto, params_dto)
        if isinstance(params_dto, AIBaseDTO):
            models = self._available_models()
            updates = {**(updates or {}), "model": models}
        return updates or {}

    # ---- actions ----
    @register_ai("Models", None)
    def list_models(self, task_dto: BaseTaskDTO, params: FilterParamsDTO | None):
        """Return a DataFrame describing discoverable AI models and metadata."""
        meta = self._model_metadata()
        rows = [
            {"model_name": name, "display_name": m.get("display_name"), "description": m.get("description")}
            for name, m in meta.items()
        ]
        return pd.DataFrame(rows, columns=["model_name", "display_name", "description"])

    @register_ai("Build Dataset", AITrainParamsDTO)
    def build_dataset_summary(self, task_dto: BaseTaskDTO, params: AITrainParamsDTO):
        """Return a one-row DataFrame summarizing dataset size and class balance."""
        X, y, meta = self._build_epoch_dataset(task_dto, params)
        if X is None:
            return pd.DataFrame({"status": [meta.get("reason", "unavailable")]})
        n_e, n_c, n_t = X.shape
        uniq, counts = np.unique(y, return_counts=True)
        dist = {f"label::{str(k)}": int(v) for k, v in zip(uniq, counts)}
        row = {
            "n_epochs": int(n_e),
            "n_channels": int(n_c),
            "n_times": int(n_t),
            "sfreq": meta.get("sfreq", 0.0),
            **dist,
        }
        return pd.DataFrame([row])

    @register_ai("Train", AITrainParamsDTO)
    def train(self, task_dto: BaseTaskDTO, params: AITrainParamsDTO):
        """Train a simple classifier and return the in-memory model (no saving)."""
        if torch is None:
            return {"status": "error", "reason": "torch not available"}
        X, y_raw, meta = self._build_epoch_dataset(task_dto, params)
        if X is None:
            return {"status": "unavailable", "reason": meta.get("reason")}
        y_idx, classes = self._encode_labels(y_raw)

        device = self._select_device(
            (params.device or ["auto"])[0] if isinstance(params.device, list) else params.device)
        X_t = torch.from_numpy(X)
        y_t = torch.from_numpy(y_idx)
        n_e, n_c, n_t = X.shape
        model, info = self._model_factory(
            (params.model or [None])[0] if isinstance(params.model, list) else params.model, n_c, n_t, len(classes))
        if model is None:
            return {"status": "unsupported", **info}
        model = model.to(device)
        adapter = info.get("input_adapter")

        ds = TensorDataset(X_t, y_t)
        dl = DataLoader(ds, batch_size=int(params.batch_size), shuffle=True)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=float(params.lr))
        model.train()
        for _ in range(int(params.epochs_n)):
            for xb, yb in dl:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                if adapter == "flatten":
                    xb_in = xb.reshape(xb.size(0), -1)
                elif adapter == "channels_times":
                    xb_in = xb
                else:
                    xb_in = xb
                out = model(xb_in)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()

        return model

    @register_ai("Predict", AIPredictParamsDTO)
    def predict(self, task_dto: BaseTaskDTO, params: AIPredictParamsDTO):
        """Return placeholder status until checkpoint-based inference is added."""
        return {
            "status": "placeholder",
            "message": "Predict action not implemented yet. Checkpoint-based inference will be added with tmux jobs.",
        }
