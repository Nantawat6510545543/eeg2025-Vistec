"""Predict placeholder (to be replaced with checkpoint-based inference)."""

from __future__ import annotations

from ...models.dtos import AIPredictParamsDTO, BaseTaskDTO
from . import register_ai


@register_ai("Predict", AIPredictParamsDTO)
def predict(self, task_dto: BaseTaskDTO, params: AIPredictParamsDTO):
    """Return placeholder status until checkpoint-based inference is added."""
    return {
        "status": "placeholder",
        "message": "Predict action not implemented yet. Checkpoint-based inference will be added with tmux jobs.",
    }
