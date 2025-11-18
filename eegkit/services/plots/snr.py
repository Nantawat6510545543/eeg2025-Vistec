from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..plots import register_plot
from ...models.dtos import BaseTaskDTO, EpochPSDParamsDTO
from ...utils.channels import prepare_channels
from ...utils.plot import finalize_figure
from ...utils.signal import snr_spectrum

plt.ioff()


@register_plot("SNR Spectrum", EpochPSDParamsDTO)
def plot_snr(self, task_dto: BaseTaskDTO, params: EpochPSDParamsDTO):
    epochs, labels = self.get_epochs(task_dto, params)
    if epochs is None:
        return None
    epochs = prepare_channels(epochs, params)
    sfreq = epochs.info["sfreq"]
    spectrum = epochs.compute_psd(
        method="welch",
        n_fft=int(max(8, sfreq * (params.tmax - params.tmin))),  # minimal n_fft safeguard
        n_overlap=0,
        n_per_seg=None,
        tmin=params.tmin,
        tmax=params.tmax,
        fmin=params.fmin,
        fmax=params.fmax,
        window="hann",
        average='mean',
        verbose=False,
    )
    psds, freqs = spectrum.get_data(return_freqs=True)
    snrs = snr_spectrum(psds)
    fig, axes = plt.subplots(2, 1, sharex="all", figsize=(8, 5))
    try:
        start_idx = int(np.where(np.floor(freqs) == 1.0)[0][0])
    except IndexError:
        start_idx = 0
    try:
        end_idx = int(np.where(np.ceil(freqs) == params.fmax - 1)[0][0])
    except IndexError:
        end_idx = len(freqs) - 1
    if end_idx <= start_idx:
        start_idx, end_idx = 0, len(freqs) - 1
    freq_idx = range(start_idx, end_idx)
    psds_db = 10 * np.log10(psds, where=psds > 0, out=np.full_like(psds, np.nan))
    psds_mean = np.nanmean(psds_db[..., freq_idx], axis=(0, 1))
    psds_std = np.nanstd(psds_db[..., freq_idx], axis=(0, 1))
    axes[0].plot(freqs[freq_idx], psds_mean, color="b")
    axes[0].fill_between(
        freqs[freq_idx], psds_mean - psds_std, psds_mean + psds_std, color="b", alpha=0.2
    )
    axes[0].set(title="PSD spectrum", ylabel="Power Spectral Density [dB]")
    snr_mean = np.nanmean(snrs[..., freq_idx], axis=(0, 1))
    snr_std = np.nanstd(snrs[..., freq_idx], axis=(0, 1))
    axes[1].plot(freqs[freq_idx], snr_mean, color="r")
    axes[1].fill_between(
        freqs[freq_idx], snr_mean - snr_std, snr_mean + snr_std, color="r", alpha=0.2
    )
    axes[1].set(
        title="SNR spectrum",
        xlabel="Frequency [Hz]",
        ylabel="SNR",
        ylim=[-2, 30],
        xlim=[params.fmin, params.fmax],
    )
    return finalize_figure(fig, task_dto, params.stimulus, caption_line=str(params), plot_name="SNR Spectrum")
