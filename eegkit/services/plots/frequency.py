from __future__ import annotations

import matplotlib.pyplot as plt

from ..plots import register_plot
from ...models.dtos import BaseTaskDTO, EpochPSDParamsDTO
from ...utils.channels import prepare_channels
from ...utils.plot import finalize_figure

plt.ioff()


@register_plot("Frequency Domain", EpochPSDParamsDTO)
def plot_frequency(self, task_dto: BaseTaskDTO, params: EpochPSDParamsDTO):
    epochs, labels = self.get_epochs(task_dto, params)
    if epochs is None:
        return None
    epochs = prepare_channels(epochs, params)
    sfreq = epochs.info["sfreq"]
    nfft = int(max(8, sfreq * max(0.5, (params.tmax - params.tmin))))
    psd = epochs.compute_psd(
        method="welch",
        fmin=params.fmin,
        fmax=params.fmax,
        tmin=params.tmin,
        tmax=params.tmax,
        n_fft=nfft,
        window="hann",
        average='mean',
    )
    fig = psd.plot(average=params.average, spatial_colors=params.spatial_colors, dB=params.dB, show=False)
    return finalize_figure(fig, task_dto, caption_line=str(params), plot_name="Frequency Domain")
