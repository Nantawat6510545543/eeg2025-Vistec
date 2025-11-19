"""Service producing structured tabular views (annotations, channels, metadata, epochs)."""

from .base_service import BaseService
from .data_views import data_registry  # populated by submodules on import


class EEGDataService(BaseService):
    """Return lightweight DataFrames summarizing current EEG selection."""

    description = "Provides structured tables from the current selection for quick inspection and lightweight export (annotations, channels/electrodes, metadata, epoch summaries)."

    def __init__(self, get_raw_func, get_epochs_func, get_task_func):
        """Initialize with controller callbacks and bind data view registry."""
        super().__init__(
            registry=data_registry,
            get_raw_func=get_raw_func,
            get_epochs_func=get_epochs_func,
            get_task_func=get_task_func,
        )
