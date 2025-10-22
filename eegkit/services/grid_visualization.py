from __future__ import annotations

import copy
import numpy as np

from .base_service import BaseService
from ..models import (
    BaseTaskDTO,
    EpochPSDParamsDTO,
    EvokedParamsDTO,
)
from ..utils import (
    render_label_grid,
    draw_evoked_response,
    prepare_channels,
)


grid_plot_registry = {}


def register_grid_plot(name, dto_cls):
    def decorator(func):
        grid_plot_registry[name] = {
            "params": dto_cls,
            "function": func,
        }
        return func

    return decorator


class EEGGridVisualization(BaseService):
    description = "Displays per-condition results in a labeled grid for side-by-side comparison with consistent axes and scaling."

    def __init__(self, get_raw_func, get_epochs_func, get_evoked_func, get_task_func=None):
        super().__init__(
            registry=grid_plot_registry,
            get_raw_func=get_raw_func,
            get_epochs_func=get_epochs_func,
            get_evoked_func=get_evoked_func,
            get_task_func=get_task_func,
        )

    @register_grid_plot("PSD Grid", EpochPSDParamsDTO)
    def plot_psd_grid(self, task_dto: BaseTaskDTO, params: EpochPSDParamsDTO):
        epochs, available_labels = self.get_epochs(task_dto, params)
        if epochs is None:
            return None

        sfreq = float(epochs.info.get("sfreq", 0.0))
        nfft = int(max(8, sfreq * max(0.5, (params.tmax - params.tmin) if (params.tmax is not None and params.tmin is not None) else 1.0)))

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
                psd_db = 10 * np.log10(psd, where=psd > 0, out=np.full_like(psd, np.nan))
                psd_mean = np.nanmean(psd_db, axis=(0, 1))
                psd_std = np.nanstd(psd_db, axis=(0, 1))
                ax.plot(freqs, psd_mean, color='b')
                ax.fill_between(freqs, psd_mean - psd_std, psd_mean + psd_std, color='b', alpha=0.2)
                ax.text(1, 1, f"n={int(len(ce))}", transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color='0.4')
                if psd_mean.size:
                    return float(np.nanmin(psd_mean)), float(np.nanmax(psd_mean))
            except Exception:
                return None

        return render_label_grid(
            task_dto=task_dto,
            epochs=epochs,
            available_labels=available_labels,
            params=params,
            plot_name="PSD Grid",
            xlim=(params.fmin, params.fmax),
            xlabel="Frequency [Hz]",
            unit_tag="dB",
            scale_mode=scale_mode,
            per_cell_draw=_draw,
        )

    @register_grid_plot("SNR Grid", EpochPSDParamsDTO)
    def plot_snr_grid(self, task_dto: BaseTaskDTO, params: EpochPSDParamsDTO):
        epochs, available_labels = self.get_epochs(task_dto, params)
        if epochs is None:
            return None

        sfreq = float(epochs.info.get("sfreq", 0.0))
        nfft = int(max(8, sfreq * max(0.5, (params.tmax - params.tmin) if (params.tmax is not None and params.tmin is not None) else 1.0)))

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
                snr = self._snr_spectrum(psd)
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
            # compute y-range
            try:
                data_uv = evoked.data * 1e6
                dmin = float(np.nanmin(data_uv)) if data_uv.size else None
                dmax = float(np.nanmax(data_uv)) if data_uv.size else None
            except Exception:
                dmin = dmax = None
            draw_evoked_response(ax, evoked, p)
            try:
                nave = getattr(evoked, 'nave', None)
                if nave is not None:
                    ax.text(1, 1, f"n={int(nave)}", transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color='0.4')
            except Exception:
                pass
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
