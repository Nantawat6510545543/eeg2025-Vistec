from __future__ import annotations

import numpy as np

from . import register_grid_plot
from ...models.dtos import BaseTaskDTO, EpochPSDParamsDTO
from ...utils.channels import prepare_channels
from ...utils.plot import render_label_grid
from ...utils.signal import snr_spectrum


@register_grid_plot("SNR Grid", EpochPSDParamsDTO)
def plot_snr_grid(self, task_dto: BaseTaskDTO, params: EpochPSDParamsDTO):
    epochs, available_labels = self.get_epochs(task_dto, params)
    if epochs is None:
        return None

    sfreq = float(epochs.info.get("sfreq", 0.0))
    nfft = int(
        max(8, sfreq * max(0.5, (params.tmax - params.tmin) if (
                params.tmax is not None and params.tmin is not None) else 1.0))
    )

    scale_mode = getattr(params, 'scale_mode', 'per-plot')
    if isinstance(scale_mode, (list, tuple)) and scale_mode:
        scale_mode = scale_mode[0]

    def _draw(ax, label):
        try:
            ce = prepare_channels(epochs[label], params)
            if len(ce) == 0:
                return None
            spectrum = ce.compute_psd(
                method="welch",
                n_fft=nfft,
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
            psd, freqs = spectrum.get_data(return_freqs=True)
            snr = snr_spectrum(psd)
            snr_mean = np.nanmean(snr, axis=(0, 1))
            snr_std = np.nanstd(snr, axis=(0, 1))
            ax.plot(freqs, snr_mean, color='r')
            ax.fill_between(freqs, snr_mean - snr_std, snr_mean + snr_std, color='r', alpha=0.2)
            ax.text(1, 1, f"n={int(len(ce))}", transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color='0.4')
            if snr_mean.size:
                return float(np.nanmin(snr_mean)), float(np.nanmax(snr_mean))
        except Exception:
            return None

    return render_label_grid(
        task_dto=task_dto,
        epochs=epochs,
        available_labels=available_labels,
        params=params,
        plot_name="SNR Grid",
        xlim=(params.fmin, params.fmax),
        xlabel="Frequency [Hz]",
        unit_tag="SNR",
        scale_mode=scale_mode,
        per_cell_draw=_draw,
    )
