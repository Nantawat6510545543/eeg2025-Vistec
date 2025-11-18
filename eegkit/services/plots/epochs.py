from __future__ import annotations

import matplotlib.pyplot as plt

from ...models.dtos import BaseTaskDTO, EpochParamsDTO
from ...utils.plot import finalize_figure
from ...utils.channels import prepare_channels
from ..plots import register_plot

plt.ioff()


@register_plot("Epoch Plot", EpochParamsDTO)
def plot_epochs(self, task_dto: BaseTaskDTO, params: EpochParamsDTO):
    epochs, labels = self.get_epochs(task_dto, params)
    if epochs is None:
        return None
    if params.stimulus and params.stimulus in epochs.event_id:
        epochs = epochs[params.stimulus]
    epochs = prepare_channels(epochs, params)
    fig = epochs.plot(events=False, show=False)
    return finalize_figure(fig, task_dto, params.stimulus, caption_line=str(params), plot_name="Epoch Plot")
