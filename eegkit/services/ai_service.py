"""AI service: dataset building, simple training, and model registry introspection."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .base_service import BaseService
from ..models.dtos import (
    BaseTaskDTO,
    EpochParamsDTO,
    AIBaseDTO,
)


logger = logging.getLogger(__name__)


from .ai_actions import ai_registry  


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

    def _build_epoch_dataset(self, task_dto: BaseTaskDTO, params: EpochParamsDTO) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
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

    def prepare_params(self, task_dto, params_dto) -> Dict[str, Any]:
        """Extend BaseService.prepare_params and inject AI model choices.

        - Retains epoch label enrichment from the superclass
        - If params belong to AIBase, add {"model": [list of model names]} for UI dropdowns
        """
        updates = super().prepare_params(task_dto, params_dto)
        if isinstance(params_dto, AIBaseDTO):
            from .ai_actions.helpers import available_models
            models = available_models()
            updates = {**(updates or {}), "model": models}
        return updates or {}