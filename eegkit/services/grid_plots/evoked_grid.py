from __future__ import annotations

import copy

import numpy as np

from . import register_grid_plot
from ...models.dtos import BaseTaskDTO, EvokedParamsDTO
from ...utils.channels import prepare_channels
from ...utils.plot import render_label_grid, draw_evoked_response


@register_grid_plot("Evoked Grid", EvokedParamsDTO)
def plot_evoked_grid(self, task_dto: BaseTaskDTO, params: EvokedParamsDTO):
    epochs, available_labels = self.get_epochs(task_dto, params)
    if epochs is None:
        return None

    scale_mode = getattr(params, 'scale_mode', 'per-plot')
    if isinstance(scale_mode, (list, tuple)) and scale_mode:
        scale_mode = scale_mode[0]

    def _draw(ax, label):
        p = copy.deepcopy(params)
        p.stimulus = [label]
        evoked = self.get_evoked(task_dto, p)
        if evoked is None:
            return None
        evoked = prepare_channels(evoked, p)
        try:
            data_uv = evoked.data * 1e6
            dmin = float(np.nanmin(data_uv)) if data_uv.size else None
            dmax = float(np.nanmax(data_uv)) if data_uv.size else None
        except Exception:
            dmin = dmax = None
        draw_evoked_response(ax, evoked, p)
        nave = getattr(evoked, 'nave', None)
        if nave is not None:
            ax.text(1, 1, f"n={int(nave)}", transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color='0.4')
        if dmin is not None and dmax is not None:
            return dmin, dmax
        return None

    return render_label_grid(
        task_dto=task_dto,
        epochs=epochs,
        available_labels=available_labels,
        params=params,
        plot_name="Evoked Grid",
        xlim=(params.tmin, params.tmax),
        xlabel="Time [s]",
        unit_tag="µV",
        scale_mode=scale_mode,
        per_cell_draw=_draw,
    )
