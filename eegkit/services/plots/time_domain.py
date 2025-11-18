from __future__ import annotations

import matplotlib.pyplot as plt

from ..plots import register_plot
from ...models.dtos import BaseTaskDTO, TimeDomainParamsDTO
from ...utils.channels import prepare_channels
from ...utils.plot import finalize_figure

plt.ioff()


@register_plot("Time Domain Plot", TimeDomainParamsDTO)
def plot_time(self, task_dto: BaseTaskDTO, params: TimeDomainParamsDTO):
    raw = self.get_raw(task_dto, params)
    raw = prepare_channels(raw, params)
    fig = raw.plot(
        duration=params.duration,
        start=params.start,
        n_channels=params.n_channels,
        scalings='auto',
        show=False,
    )
    return [finalize_figure(fig, task_dto, caption_line=str(params), plot_name="Time Domain")]
