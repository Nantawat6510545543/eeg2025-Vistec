from __future__ import annotations

import matplotlib.pyplot as plt

from ..plots import register_plot
from ...models.dtos import BaseTaskDTO, EvokedParamsDTO, EvokedTopoParamsDTO, EvokedJointParamsDTO
from ...utils.channels import prepare_channels
from ...utils.plot import finalize_figure

plt.ioff()


@register_plot("Evoked Plot", EvokedParamsDTO)
def plot_evoked(self, task_dto: BaseTaskDTO, params: EvokedParamsDTO):
    evoked = self.get_evoked(task_dto, params)
    if evoked is None:
        return None
    evoked = prepare_channels(evoked, params)
    fig = evoked.plot(gfp=params.gfp, spatial_colors=params.spatial_colors, show=False)
    return finalize_figure(fig, task_dto, params.stimulus, caption_line=str(params), plot_name="Evoked Plot")


@register_plot("Evoked Topo Plot", EvokedTopoParamsDTO)
def plot_evoked_topo(self, task_dto: BaseTaskDTO, params: EvokedTopoParamsDTO):
    params.combine_channels = False
    evoked = self.get_evoked(task_dto, params)
    if evoked is None:
        return None
    evoked = prepare_channels(evoked, params)
    fig = evoked.plot_topomap(times=params.get_times, show=False)
    return finalize_figure(fig, task_dto, params.stimulus, caption_line=str(params), plot_name="Evoked Topo")


@register_plot("Evoked Plot Joint", EvokedJointParamsDTO)
def plot_evoked_joint(self, task_dto: BaseTaskDTO, params: EvokedJointParamsDTO):
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
        fig = evk.plot(gfp=copy_params.gfp, spatial_colors=copy_params.spatial_colors, show=False)
        fig = finalize_figure(
            fig,
            task_dto,
            condition,
            caption_line=str(copy_params),
            plot_name="Evoked per Condition",
        )
        fig_list.append(fig)
    return fig_list
