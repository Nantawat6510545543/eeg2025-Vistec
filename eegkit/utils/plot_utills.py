"""Small plotting helpers used by EEG visualization.

- split_tokens: split condition labels by '_'.
- compute_axes_values: infer (page, col, row) token values for grid dimensions.
- map_cells_to_labels: map a (page, col, row) triple to a concrete label.
- reshape_axes_array: normalize plt.subplots return to a 2D array.
- draw_evoked_response: draw per-channel traces, optional average line, and GFP with reference lines.
"""
from __future__ import annotations

import numpy as np


def split_tokens(label: str) -> list[str]:
    return [token for token in (label or '').split('_') if token]


def compute_axes_values(tokens_by_label: dict[str, list[str]], mode: int):
    axes_values = []
    for token_index in range(mode):
        values = sorted({tokens[token_index] for tokens in tokens_by_label.values() if len(tokens) > token_index})
        axes_values.append(values if values else [None])
    while len(axes_values) < 3:
        axes_values.insert(0, [None])
    return tuple(axes_values)


def map_cells_to_labels(tokens_by_label: dict[str, list[str]], mode: int):
    mapping = {}
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
    if num_rows == 1 and num_cols == 1:
        return np.array([[axes]])
    if num_rows == 1:
        return np.array([axes])
    if num_cols == 1:
        return np.array([[axis] for axis in axes])
    return axes


def draw_evoked_response(axis, evoked, params):
    """Draw evoked time series onto a Matplotlib axis.

    - Per-channel traces (grey) unless GFP is set to "only".
    - Optional channel-average line (orange) when params.average_line is True.
    - GFP in black when params.gfp is True or "only".
    - Vertical (t=0) and horizontal (y=0) reference lines.
    """
    times = evoked.times
    data_microvolts = evoked.data * 1e6

    # Per-channel traces (skip if GFP-only)
    if getattr(params, 'gfp', None) != "only":
        for ch in range(data_microvolts.shape[0]):
            axis.plot(times, data_microvolts[ch], color='0.75', linewidth=0.6, zorder=1)

    # Optional average line across channels
    if getattr(params, 'average_line', False):
        if data_microvolts.ndim == 2 and data_microvolts.shape[0] > 1:
            mean_signal = np.nanmean(data_microvolts, axis=0)
        else:
            mean_signal = data_microvolts[0] if data_microvolts.ndim == 2 else np.asarray(data_microvolts)
        axis.plot(times, mean_signal, color='tab:orange', linewidth=1.2, zorder=2.4, alpha=0.95)

    # GFP
    if getattr(params, 'gfp', None) is True or getattr(params, 'gfp', None) == "only":
        gfp_signal = np.std(data_microvolts, axis=0) if data_microvolts.shape[0] > 1 else data_microvolts[0]
        axis.plot(times, gfp_signal, color='k', linewidth=1.2, zorder=2.6)

    # Reference lines
    axis.axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.6)
    axis.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.6)
