"""Grid visualization service presenting results in a labeled matrix layout."""

from __future__ import annotations

from .base_service import BaseService
from .grid_plots import grid_plot_registry  # populated by submodules on import


class EEGGridVisualization(BaseService):
    """Service producing grid-based figures with consistent axes and scaling."""
    
    description = "Displays per-condition results in a labeled grid for side-by-side comparison with consistent axes and scaling."

    def __init__(self, get_raw_func, get_epochs_func, get_evoked_func, get_task_func=None):
        """Initialize with controller accessors and bind grid plot registry."""
        super().__init__(
            registry=grid_plot_registry,
            get_raw_func=get_raw_func,
            get_epochs_func=get_epochs_func,
            get_evoked_func=get_evoked_func,
            get_task_func=get_task_func,
        )
