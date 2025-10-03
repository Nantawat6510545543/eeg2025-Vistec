"""Small plotting helpers used by EEG visualization.

- split_tokens: split condition labels by '_'.
- compute_axes_values: infer (page, col, row) token values for grid dimensions.
- map_cells_to_labels: map a (page, col, row) triple to a concrete label.
- reshape_axes_array: normalize plt.subplots return to a 2D array.
- draw_evoked_response: draw per-channel traces, optional average line, and GFP with reference lines.
"""
from __future__ import annotations

import numpy as np
import mne
from typing import List, Tuple


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


class ChannelsHelper:
    """Utilities for channel selection and optional µV range filtering.

    Behaves like EEGCleaner: constructed with params and inst, uses self.inst/self.params,
    and maintains internal state (self.picks, self.pick_names).
    """

    def __init__(self, params, inst):
        self.params = params
        self.inst = inst
        self.picks: List[int] | None = None
        self.pick_names: List[str] | None = None

    def pick_channels(self) -> None:
        ch = getattr(self.params, 'channels_list', []) or []
        # Respect showbad: exclude marked bads from candidates unless requested
        if getattr(self.params, 'showbad', False):
            exclude = []
        else:
            exclude = list(getattr(self.inst.info, 'bads', []) or [])
        picks_raw = mne.pick_channels(self.inst.ch_names, include=ch, exclude=exclude)
        try:
            picks = [int(i) for i in np.array(picks_raw).tolist()]
        except Exception:
            picks = list(picks_raw) if isinstance(picks_raw, (list, tuple)) else []
        self.picks = picks
        self.pick_names = [self.inst.ch_names[i] for i in picks]

    def filter_by_uv(self) -> None:
        # Coerce uv_min/uv_max to floats or None (UI may provide empty strings)
        def _to_float_or_none(x):
            if x is None:
                return None
            try:
                if isinstance(x, str) and x.strip() == "":
                    return None
                return float(x)
            except Exception:
                return None

        uv_min = _to_float_or_none(getattr(self.params, 'uv_min', None))
        uv_max = _to_float_or_none(getattr(self.params, 'uv_max', None))

        if self.picks is None:
            self.pick_channels()

        picks = self.picks or []

        if not (uv_min is not None or uv_max is not None):
            # nothing to filter
            return

        # Robust empty check
        if picks is None or (hasattr(picks, "__len__") and len(picks) == 0):
            self.picks = []
            self.pick_names = []
            return

        # Determine data array in µV for the selected picks
        inst = self.inst
        data_uv = None
        try:
            if hasattr(inst, 'data') and not hasattr(inst, 'get_data'):
                data_uv = inst.data[picks, :] * 1e6
            else:
                arr = inst.get_data(picks=picks)
                if arr.ndim == 2:
                    data_uv = arr * 1e6  # (n_ch, n_times)
                elif arr.ndim == 3:
                    n_epochs, n_ch, n_times = arr.shape
                    data_uv = arr.transpose(1, 0, 2).reshape(n_ch, -1) * 1e6
                else:
                    data_uv = None
        except Exception:
            data_uv = None

        if data_uv is None:
            # cannot determine, leave picks unchanged
            return

        ch_mins = np.nanmin(data_uv, axis=-1)
        ch_maxs = np.nanmax(data_uv, axis=-1)
        keep_mask = np.ones(len(picks), dtype=bool)
        if uv_min is not None:
            keep_mask &= ch_mins >= uv_min
        if uv_max is not None:
            keep_mask &= ch_maxs <= uv_max

        kept = [idx for idx, keep in zip(picks, keep_mask) if keep]
        # If filter removes everything, fall back to original picks to avoid empty selection
        final_picks = kept if kept else picks
        self.picks = final_picks
        self.pick_names = [self.inst.ch_names[i] for i in final_picks]
