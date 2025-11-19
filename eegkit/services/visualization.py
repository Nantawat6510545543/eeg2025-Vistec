"""Single-figure visualization service for current EEG selection (epochs/raw/evoked)."""

from .base_service import BaseService
from .plots import plot_registry  # populated by submodules on import


class EEGVisualization(BaseService):
    """Produce one concise figure for a task (frequency/SNR/evoked/time-domain etc.)."""

    description = "Produces one concise figure for the current EEG selection."

    def __init__(self, get_raw_func, get_epochs_func, get_task_func, get_evoked_func):
        """Initialize with controller callbacks and bind plot registry."""
        super().__init__(
            registry=plot_registry,
            get_raw_func=get_raw_func,
            get_epochs_func=get_epochs_func,
            get_evoked_func=get_evoked_func,
            get_task_func=get_task_func,
        )
