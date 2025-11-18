"""Grid/token plotting helpers used by EEG visualization.

Provides helpers for:
- Token splitting (condition label parsing)
- Axis value inference for page/column/row dimensions
- Cell→label mapping
- Axes reshaping
- Evoked trace rendering (per-channel, average, GFP)
- Generic label-tokenized grid rendering
"""
from __future__ import annotations

from typing import Tuple, Callable, Optional, Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from tqdm.auto import tqdm

from .figure_utils import finalize_figure


def split_tokens(label: str) -> list[str]:
    """Split a label by underscores and drop empty tokens."""
    return [token for token in (label or '').split('_') if token]


def compute_axes_values(tokens_by_label: dict[str, list[str]], mode: int) -> tuple[
    list[Optional[str]], list[Optional[str]], list[Optional[str]]]:
    """Compute possible values for page/column/row axes based on tokens and mode."""
    axes_values = []
    for token_index in range(mode):
        values = sorted({tokens[token_index] for tokens in tokens_by_label.values() if len(tokens) > token_index})
        axes_values.append(values if values else [None])
    while len(axes_values) < 3:
        axes_values.insert(0, [None])
    return tuple(axes_values)


def map_cells_to_labels(tokens_by_label: dict[str, list[str]], mode: int) -> Dict[
    Tuple[Optional[str], Optional[str], Optional[str]], str]:
    """Map each (page, column, row) triple to the first matching label."""
    mapping: Dict[Tuple[Optional[str], Optional[str], Optional[str]], str] = {}
    for label, tokens in tokens_by_label.items():
        if len(tokens) >= mode:
            if mode == 1:
                key = (None, None, tokens[0])
            elif mode == 2:
                key = (None, tokens[0], tokens[1])
            else:
                key = (tokens[0], tokens[1], tokens[2])
            mapping.setdefault(key, label)  # first-come deterministic
    return mapping


def reshape_axes_array(axes, num_rows: int, num_cols: int):
    """Normalize a Matplotlib axes return value to a 2D array shape."""
    if num_rows == 1 and num_cols == 1:
        return np.array([[axes]])
    if num_rows == 1:
        return np.array([axes])
    if num_cols == 1:
        return np.array([[axis] for axis in axes])
    return axes


def draw_evoked_response(axis, evoked, params):
    """Draw evoked time series onto a Matplotlib axis."""
    times = evoked.times
    data_microvolts = evoked.data * 1e6

    if getattr(params, 'gfp', None) != "only":
        for ch in range(data_microvolts.shape[0]):
            axis.plot(times, data_microvolts[ch], color='0.75', linewidth=0.6, zorder=1)

    if getattr(params, 'average_line', False):
        if data_microvolts.ndim == 2 and data_microvolts.shape[0] > 1:
            mean_signal = np.nanmean(data_microvolts, axis=0)
        else:
            mean_signal = data_microvolts[0] if data_microvolts.ndim == 2 else np.asarray(data_microvolts)
        axis.plot(times, mean_signal, color='tab:orange', linewidth=1.2, zorder=2.4, alpha=0.95)

    if getattr(params, 'gfp', None) is True or getattr(params, 'gfp', None) == "only":
        gfp_signal = np.std(data_microvolts, axis=0) if data_microvolts.shape[0] > 1 else data_microvolts[0]
        axis.plot(times, gfp_signal, color='k', linewidth=1.2, zorder=2.6)

    axis.axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.6)
    axis.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.6)


def render_label_grid(
        *,
        task_dto,
        epochs,
        available_labels,
        params,
        plot_name: str,
        xlim: tuple[float, float],
        xlabel: str,
        unit_tag: str,
        scale_mode: str,
        per_cell_draw: Callable[[Axes, str], Optional[Tuple[float, float]]],
):
    """Render label-tokenized grids."""
    tokens_by_label = {label: split_tokens(label) for label in available_labels}
    max_token_count = max((len(tokens) for tokens in tokens_by_label.values()), default=1)
    grid_mode = min(max_token_count, 3)
    page_values, column_values, row_values = compute_axes_values(tokens_by_label, grid_mode)
    cell_to_label_map = map_cells_to_labels(tokens_by_label, grid_mode)

    figures = []
    num_rows, num_cols = len(row_values), len(column_values)
    total_cells = len(page_values) * max(1, num_rows) * max(1, num_cols)

    with tqdm(total=total_cells, desc=f"{plot_name} cells", leave=False) as pbar:
        for page_token in page_values:
            fig, axes = plt.subplots(max(1, num_rows), max(1, num_cols), sharex=False, sharey=False)
            axes_2d = reshape_axes_array(axes, max(1, num_rows), max(1, num_cols))

            y_min, y_max = None, None

            for r_idx, row_token in enumerate(row_values):
                for c_idx, col_token in enumerate(column_values):
                    ax = axes_2d[r_idx, c_idx]
                    ax.cla()

                    label = cell_to_label_map.get((page_token, col_token, row_token))
                    if label is not None and label in epochs.event_id:
                        y_bounds = per_cell_draw(ax, label)
                        if y_bounds is not None:
                            dmin, dmax = y_bounds
                            if (dmin is not None) and np.isfinite(dmin):
                                y_min = dmin if y_min is None else min(y_min, float(dmin))
                            if (dmax is not None) and np.isfinite(dmax):
                                y_max = dmax if y_max is None else max(y_max, float(dmax))

                    if r_idx == 0 and (col_token is not None and col_token != ""):
                        ax.set_title(col_token)
                    if c_idx == 0:
                        ax.set_ylabel(f"{row_token}")
                        ax.tick_params(labelleft=True)
                    else:
                        if scale_mode == 'per-plot':
                            ax.tick_params(labelleft=True)
                        else:
                            ax.tick_params(labelleft=False)

                    ax.text(0.01, 1, unit_tag, transform=ax.transAxes, ha='left', va='bottom', fontsize=8, color='0.4')
                    ax.set_xlim(*xlim)

                    pbar.update(1)

            if scale_mode == 'uniform-grid' and y_min is not None and y_max is not None:
                pad = 0.05 * max(1.0, abs(y_max - y_min))
                y_lo, y_hi = y_min - pad, y_max + pad
                for r in range(num_rows):
                    for c in range(num_cols):
                        axes_2d[r, c].set_ylim(y_lo, y_hi)

            last_row_idx = max(1, num_rows) - 1
            for r in range(num_rows):
                for c in range(num_cols):
                    ax = axes_2d[r, c]
                    if r == last_row_idx:
                        ax.set_xlabel(xlabel)
                        ax.tick_params(labelbottom=True)
                    else:
                        ax.tick_params(labelbottom=False)

            page_stimulus = page_token if (grid_mode == 3 and page_token is not None) else None
            fig = finalize_figure(
                fig,
                task_dto,
                stimulus=page_stimulus,
                caption_line=str(params),
                plot_name=plot_name,
            )
            figures.append(fig)

    return figures

__all__ = [
    "split_tokens",
    "compute_axes_values",
    "map_cells_to_labels",
    "reshape_axes_array",
    "draw_evoked_response",
    "render_label_grid",
]
