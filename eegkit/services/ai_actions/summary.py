"""Model summary action (parameter counts and layer outputs)."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore

from ...models.dtos import EEGNetMultiRegTrainParamsDTO, BaseTaskDTO
from . import register_ai
from ..ai_models import EEGNetMultiReg
from .helpers import model_metadata


@register_ai("Model Summary", EEGNetMultiRegTrainParamsDTO)
def model_summary(self, task_dto: BaseTaskDTO, params: EEGNetMultiRegTrainParamsDTO) -> str:
    """Return a torchinfo-style textual summary for EEGNetMultiReg."""
    if torch is None:
        return "torch not available"
    X, y_raw, meta = self._build_epoch_dataset(task_dto, params)
    if X is None:
        return f"unavailable: {meta.get('reason')}"
    n_e, n_c, n_t = X.shape
    model = EEGNetMultiReg(n_outputs=1)
    input_size = (1, n_c, n_t)  # batch=1 for summary
    # torchinfo may not be installed; try import lazily here
    try:
        from torchinfo import summary as torchinfo_summary  # type: ignore
        info = torchinfo_summary(model, input_size=input_size, col_names=("output_size", "num_params"), verbose=2)
        return str(info)
    except Exception:
        return "torchinfo not installed or summary failed"
