from __future__ import annotations

from abc import ABC
from typing import Optional, Dict, Any
import numpy as np

# Models are needed for dynamic param preparation
from ..models import EpochParamsDTO


class BaseService(ABC):
    """Abstract base for controller modes (plot / grid_plot / data).

    Responsibilities:
    - Hold human-readable description for UI
    - Build bound spec from a provided registry (name -> {"params", "function"})
    - Provide shared helpers like prepare_params()

    Subclasses should provide a REGISTRY (dict) or pass one into __init__.
    """

    # Human-readable mode description for the UI
    description: str = ""

    def __init__(
        self,
        *,
        registry: Dict[str, Dict[str, Any]],
        get_raw_func=None,
        get_epochs_func=None,
        get_evoked_func=None,
        get_task_func=None,
    ):
        # Wire controller accessors (subclasses may use a subset)
        self.get_raw = get_raw_func
        self.get_epochs = get_epochs_func
        self.get_evoked = get_evoked_func
        self.get_task = get_task_func

        # Bind registry functions to this instance
        self.spec = registry or {}
        for key in list(self.spec.keys()):
            func = self.spec[key]["function"]
            # Bind the unbound function to this instance
            self.spec[key]["function"] = func.__get__(self)  # type: ignore[attr-defined]

    # ----- Shared helpers -----

    def prepare_params(self, task_dto, params_dto):
        if EpochParamsDTO and isinstance(params_dto, EpochParamsDTO):
            if self.get_epochs is None:
                return {}
            _epochs, labels = self.get_epochs(task_dto=task_dto, epoch_params=params_dto)
            if isinstance(labels, str) and labels == "unavailable":
                return {}
            if labels is not None:
                return {"stimulus": [None] + list(labels)}
        return {}
