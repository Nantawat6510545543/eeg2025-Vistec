"""Evoked-response plot functions (single, topo, joint, per-condition)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..plots import register_plot
from ...models.dtos import BaseTaskDTO, EvokedParamsDTO, EvokedTopoParamsDTO, EvokedJointParamsDTO
from ...utils.channels import prepare_channels
from ...utils.plot import finalize_figure
from ..grid_plots.helpers import draw_evoked_response

plt.ioff()


@register_plot("Evoked Plot", EvokedParamsDTO)
def plot_evoked(self, task_dto: BaseTaskDTO, params: EvokedParamsDTO):
    """Plot evoked time course; return finalized figure."""
    evoked = self.get_evoked(task_dto, params)
    if evoked is None:
        return None
    evoked = prepare_channels(evoked, params)

    # Use custom renderer to support average line + error band.
    fig, ax = plt.subplots(1, 1)
    draw_evoked_response(ax, evoked, params)
    x_lo = params.display_tmin if getattr(params, 'display_tmin', None) is not None else params.tmin
    x_hi = params.display_tmax if getattr(params, 'display_tmax', None) is not None else params.tmax
    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("µV")
    nave = getattr(evoked, 'nave', None)
    if nave is not None:
        ax.text(1, 1, f"n={int(nave)}", transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color='0.4')
    return finalize_figure(fig, task_dto, params.stimulus, caption_line=str(params), plot_name="Evoked Plot")


@register_plot("Evoked Topo Plot", EvokedTopoParamsDTO)
def plot_evoked_topo(self, task_dto: BaseTaskDTO, params: EvokedTopoParamsDTO):
    """Plot evoked topomap at selected times; return finalized figure."""
    params.combine_channels = False
    evoked = self.get_evoked(task_dto, params)
    if evoked is None:
        return None
    evoked = prepare_channels(evoked, params)
    fig = evoked.plot_topomap(times=params.get_times, show=False)
    return finalize_figure(fig, task_dto, params.stimulus, caption_line=str(params), plot_name="Evoked Topo")


@register_plot("Evoked Plot Joint", EvokedJointParamsDTO)
def plot_evoked_joint(self, task_dto: BaseTaskDTO, params: EvokedJointParamsDTO):
    """Plot joint time course + topomap panels; return finalized figure."""
    params.combine_channels = False
    evoked = self.get_evoked(task_dto, params)
    if evoked is None:
        return None
    evoked = prepare_channels(evoked, params)
    fig = evoked.plot_joint(
        times=params.get_times,
        topomap_args={},
        ts_args={"gfp": params.gfp, "spatial_colors": params.spatial_colors},
        show=False,
    )
    return finalize_figure(fig, task_dto, params.stimulus, caption_line=str(params), plot_name="Evoked Joint")


@register_plot("Evoked per Condition", EvokedParamsDTO)
def plot_evoked_per_condition(self, task_dto: BaseTaskDTO, params: EvokedParamsDTO):
    """Return list of evoked figures, one per available condition/label."""
    import copy as _copy

    epochs, labels = self.get_epochs(task_dto, params)
    if epochs is None:
        return None
    fig_list = []
    for condition in epochs.event_id:
        copy_params = _copy.deepcopy(params)
        copy_params.stimulus = condition
        evk = self.get_evoked(task_dto, copy_params)
        if evk is None:
            continue
        evk = prepare_channels(evk, copy_params)
        fig, ax = plt.subplots(1, 1)
        draw_evoked_response(ax, evk, copy_params)
        x_lo = copy_params.display_tmin if getattr(copy_params, 'display_tmin', None) is not None else copy_params.tmin
        x_hi = copy_params.display_tmax if getattr(copy_params, 'display_tmax', None) is not None else copy_params.tmax
        ax.set_xlim(x_lo, x_hi)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("µV")
        nave = getattr(evk, 'nave', None)
        if nave is not None:
            ax.text(1, 1, f"n={int(nave)}", transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color='0.4')
        fig = finalize_figure(
            fig,
            task_dto,
            condition,
            caption_line=str(copy_params),
            plot_name="Evoked per Condition",
        )
        fig_list.append(fig)
    return fig_list
