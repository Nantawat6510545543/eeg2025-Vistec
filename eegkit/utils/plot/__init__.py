"""Plotting helpers category.

Re-exports visualization-related utilities from the utils package so call sites
can use a discoverable import path:

    from eegkit.utils.plot import (
        finalize_figure,
        split_tokens, compute_axes_values, map_cells_to_labels,
        reshape_axes_array, draw_evoked_response, render_label_grid,
    )

This is a thin layer; the original modules remain available for backward
compatibility.
"""

from .figure_utils import finalize_figure
from .plot_utils import (
    split_tokens,
    compute_axes_values,
    map_cells_to_labels,
    reshape_axes_array,
    draw_evoked_response,
    render_label_grid,
)

__all__ = [
    "finalize_figure",
    "split_tokens",
    "compute_axes_values",
    "map_cells_to_labels",
    "reshape_axes_array",
    "draw_evoked_response",
    "render_label_grid",
]
