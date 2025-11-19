"""Build a simple dataset summary for training inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...models.dtos import AITrainParamsDTO, BaseTaskDTO
from . import register_ai


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
