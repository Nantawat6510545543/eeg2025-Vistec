from __future__ import annotations

from . import register_data
from ...models.dtos import BaseTaskDTO, FilterParamsDTO


@register_data("Metadata", None)
def show_annotations(self, task_dto: BaseTaskDTO, params: FilterParamsDTO):
    task_model = self.get_task(task_dto)
    return task_model.metadata
