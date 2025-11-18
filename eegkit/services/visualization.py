from .base_service import BaseService
from .plots import plot_registry  # populated by submodules on import


class EEGVisualization(BaseService):
    description = "Produces one concise figure for the current EEG selection."

    def __init__(self, get_raw_func, get_epochs_func, get_task_func, get_evoked_func):
        super().__init__(
            registry=plot_registry,
            get_raw_func=get_raw_func,
            get_epochs_func=get_epochs_func,
            get_evoked_func=get_evoked_func,
            get_task_func=get_task_func,
        )
