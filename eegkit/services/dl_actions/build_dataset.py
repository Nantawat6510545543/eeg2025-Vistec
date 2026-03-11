from __future__ import annotations

from ...models.dtos import DLTrainParamsDTO, BaseTaskDTO
from . import register_dl


@register_dl("Build Dataset", DLTrainParamsDTO)
def build_dataset(self, task_dto: BaseTaskDTO, params: DLTrainParamsDTO):
    pass
