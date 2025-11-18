"""Channel selection and optional µV range filtering utilities.

This module provides ChannelsHelper used by visualization utilities to:
- pick channels based on params.channels_list and showbad flag,
- optionally filter channels by complete-trace microvolt (µV) ranges,
- expose `picks` and `pick_names` for subsequent processing.
"""
from __future__ import annotations

from typing import List

import mne
import numpy as np


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


def prepare_channels(inst, params):
    """End-to-end channel preparation as a simple module function.

    - For Epochs-like objects, ensure data is loaded (copy.load_data()).
    - Select channels according to params (including showbad behavior).
    - Optionally filter channels by µV range across the full trace.
    - Optionally combine selected channels into a single 'combined' channel.

    Returns a new instance (picked or channel-combined), leaving the input unmodified.
    """
    # Ensure data is loaded for objects that support lazy loading (e.g., Epochs)
    if hasattr(inst, "event_id") and hasattr(inst, "load_data"):
        inst = inst.copy().load_data()

    helper = ChannelsHelper(params, inst)
    helper.pick_channels()
    helper.filter_by_uv()
    picks = helper.picks or []
    pick_names = helper.pick_names or []

    if getattr(params, 'combine_channels', False):
        if pick_names:
            return mne.channels.combine_channels(
                inst, groups={"combined": list(pick_names)}, method="mean"
            )
        return inst.copy()
    else:
        return inst.copy().pick(picks)
