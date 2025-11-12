from __future__ import annotations

"""AI Service and Mediator

Provides a registry-driven AI mode similar to plot/data services and a thin
mediator that adapts EEG epochs into AI-friendly tensors. Training is kept
minimal for now; we focus on wiring, parameter preparation, and dataset build.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base_service import BaseService
from ..models import (
    BaseTaskDTO,
    EpochParamsDTO,
    FilterParamsDTO,
    AIBaseDTO,
    AITrainParamsDTO,
    AIPredictParamsDTO,
)

ai_registry: Dict[str, Dict[str, Any]] = {}


def register_ai(name: str, dto_cls):
    def _decorator(func):
        ai_registry[name] = {"params": dto_cls, "function": func}
        return func

    return _decorator

class EEGAIService(BaseService):
    description = "AI training and inference on epochs (registry-based)."

    def __init__(self, *, get_raw_func=None, get_epochs_func=None, get_task_func=None):
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
            from importlib import import_module
            mod = import_module('eegkit.services.ai_models')
            import torch.nn as nn
            for attr in getattr(mod, '__all__', []):
                obj = getattr(mod, attr, None)
                if isinstance(obj, type) and issubclass(obj, nn.Module):
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

        try:
            epochs.load_data()
        except Exception:
            pass

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
        meta = self._model_metadata()
        rows = [
            {"model_name": name, "display_name": m.get("display_name"), "description": m.get("description")}
            for name, m in meta.items()
        ]
        return pd.DataFrame(rows, columns=["model_name", "display_name", "description"])

    @register_ai("Build Dataset", AITrainParamsDTO)
    def build_dataset_summary(self, task_dto: BaseTaskDTO, params: AITrainParamsDTO):
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
    def train_stub(self, task_dto: BaseTaskDTO, params: AITrainParamsDTO):
        # Placeholder: just verify data can be built
        X, y, meta = self._build_epoch_dataset(task_dto, params)
        if X is None:
            return {"status": "unavailable", "reason": meta.get("reason")}
        return {
            "status": "ok",
            "message": "Training pipeline scaffolded. Replace with real training.",
            "shape": meta.get("shape"),
        }
