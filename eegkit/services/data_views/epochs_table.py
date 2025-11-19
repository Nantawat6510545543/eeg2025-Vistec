"""Tabular summary for epoch collections.

Provides a simple per-condition table with counts, sampling rate, and durations.
"""
from __future__ import annotations

import pandas as pd

from . import register_data
from ...models.dtos import BaseTaskDTO, EpochParamsDTO


@register_data("Epochs Table", EpochParamsDTO)
def show_epochs_table(self, task_dto: BaseTaskDTO, params: EpochParamsDTO):
    """Return a per-condition summary DataFrame for the selected epochs.

    Columns include label, number of epochs/channels, sampling rate and durations.
    """
    epochs, labels = self.get_epochs(task_dto, params)
    if epochs is None:
        return None

    rows = []
    for label, _code in epochs.event_id.items():
        try:
            cond_epochs = epochs[label]
        except Exception:
            continue
        if len(cond_epochs) == 0:
            continue
        n_times = len(cond_epochs.times)
        sfreq = float(cond_epochs.info.get('sfreq', 0.0))
        row = {
            'label': label,
            'n_epochs': len(cond_epochs),
            'n_channels': len(cond_epochs.ch_names),
            'timespan_sec': float(cond_epochs.times[-1] - cond_epochs.times[0]) if n_times > 1 else 0.0,
            'sampling_rate': sfreq,
            'duration_per_epoch_sec': float(n_times / sfreq) if sfreq > 0 and n_times > 0 else 0.0,
        }
        rows.append(row)

    return pd.DataFrame(rows)
