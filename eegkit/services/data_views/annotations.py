from __future__ import annotations

import pandas as pd

from ...models import BaseTaskDTO, FilterParamsDTO
from . import register_data


@register_data("Annotations", FilterParamsDTO)
def get_annotation_df(self, task_dto: BaseTaskDTO, filter_params: FilterParamsDTO):
    raw = self.get_raw(task_dto, filter_params)
    annots = raw.annotations
    df = pd.DataFrame({
        "onset": annots.onset,
        "duration": annots.duration,
        "description": annots.description
    })
    return df
