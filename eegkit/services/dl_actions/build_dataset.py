from __future__ import annotations

from ...models.dtos import AITrainParamsDTO, BaseTaskDTO
from . import register_dl


@register_dl("Build Dataset", AITrainParamsDTO)
def build_dataset(self, task_dto: BaseTaskDTO, params: AITrainParamsDTO):
    pass
