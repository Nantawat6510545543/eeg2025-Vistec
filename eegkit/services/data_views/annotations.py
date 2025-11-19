"""Data view for raw annotations.

Exposes onset, duration, and description from the underlying MNE annotations.
"""
from __future__ import annotations

import pandas as pd

from . import register_data
from ...models.dtos import BaseTaskDTO, FilterParamsDTO


@register_data("Annotations", FilterParamsDTO)
def get_annotation_df(self, task_dto: BaseTaskDTO, filter_params: FilterParamsDTO):
    """Return a DataFrame with annotation onset, duration, and description."""
    raw = self.get_raw(task_dto, filter_params)
    annots = raw.annotations
    df = pd.DataFrame({
        "onset": annots.onset,
        "duration": annots.duration,
        "description": annots.description
    })
    return df
