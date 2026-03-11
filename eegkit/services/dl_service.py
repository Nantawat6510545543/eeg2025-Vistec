"""AI service: dataset building, simple training, and model registry introspection."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List

import numpy as np

from .base_service import BaseService
from ..utils.channels import prepare_channels
from ..models.dtos import (
    BaseTaskDTO,
    EpochParamsDTO,
    SubjectFilterDTO,
)

from .dl_actions import dl_registry  


class EEGDLService(BaseService):
    """Provide Deep learning-related actions"""

    description = "Deep learning training and inference on epochs (registry-based)."

    def __init__(self, *, get_raw_func=None, get_epochs_func=None, get_task_func=None, get_subjects_metadata_func=None):
        """Initialize with controller callbacks and bind DL registry to spec."""
        super().__init__(
            registry=dl_registry,
            get_raw_func=get_raw_func,
            get_epochs_func=get_epochs_func,
            get_task_func=get_task_func,
        )
        self.get_subjects_metadata = get_subjects_metadata_func
