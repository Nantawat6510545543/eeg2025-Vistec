from __future__ import annotations

import matplotlib.pyplot as plt

from ...models.dtos import BaseTaskDTO, FilterParamsDTO
from ..plots import register_plot

plt.ioff()


@register_plot("Sensor Layout", FilterParamsDTO)
def plot_sensors(self, task_dto: BaseTaskDTO, params: FilterParamsDTO):
    raw = self.get_raw(task_dto, params)
    raw.pick(params.channels_list)
    fig = raw.plot_sensors(show_names=True)
    return [fig]
