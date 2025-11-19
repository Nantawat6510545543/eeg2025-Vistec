"""Spectral metrics helpers for EEG arrays."""
from __future__ import annotations

import numpy as np


def snr_spectrum(
        psd: np.ndarray,
        noise_n_neighbor_freqs: int = 3,
        noise_skip_neighbor_freqs: int = 1,
) -> np.ndarray:
    """Compute SNR by dividing PSD by a neighborhood-averaged noise estimate."""
    kernel = np.concatenate(
        (
            np.ones(noise_n_neighbor_freqs),
            np.zeros(2 * noise_skip_neighbor_freqs + 1),
            np.ones(noise_n_neighbor_freqs),
        )
    )
    kernel /= kernel.sum()
    mean_noise = np.apply_along_axis(
        lambda psd_: np.convolve(psd_, kernel, mode="valid"),
        axis=-1,
        arr=psd,
    )
    edge_width = noise_n_neighbor_freqs + noise_skip_neighbor_freqs
    pad = [(0, 0)] * (mean_noise.ndim - 1) + [(edge_width, edge_width)]
    mean_noise = np.pad(mean_noise, pad_width=tuple(pad), constant_values=float("nan"))
    return psd / mean_noise

__all__ = ["snr_spectrum"]
