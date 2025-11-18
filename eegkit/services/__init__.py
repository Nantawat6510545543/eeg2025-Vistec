"""Services package.

Visualization and data-facing services for the EEG Workbench:
- visualization: single-figure plots from epochs/evokeds
- grid_visualization: grid-based plots for per-condition comparisons
- data_service: tabular data views and lightweight exports
- plots, grid_plots, data_views: registries of concrete actions
"""

from . import data_service as data_service  # noqa: F401
from . import grid_plots as grid_plots  # noqa: F401
from . import grid_visualization as grid_visualization  # noqa: F401
from . import plots as plots  # noqa: F401
from . import visualization as visualization  # noqa: F401

__all__ = [
    'visualization',
    'data_service',
    'grid_visualization',
    'plots',
    'grid_plots',
]
