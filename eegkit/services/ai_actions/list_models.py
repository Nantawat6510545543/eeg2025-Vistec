"""List discoverable AI models and metadata."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from ...models.dtos import BaseTaskDTO, FilterParamsDTO
from . import register_ai
from .helpers import model_metadata


@register_ai("Models", None)
def list_models(self, task_dto: BaseTaskDTO, params: Optional[FilterParamsDTO]):
    """Return a DataFrame describing discoverable AI models and metadata."""
    meta = model_metadata()
    rows = [
        {"model_name": name, "display_name": m.get("display_name"), "description": m.get("description")}
        for name, m in meta.items()
    ]
    return pd.DataFrame(rows, columns=["model_name", "display_name", "description"])
