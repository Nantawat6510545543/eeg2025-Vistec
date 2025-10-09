from ..models import *
from ..utils import  (finalize_figure,
    split_tokens,
    compute_axes_values,
    map_cells_to_labels,
    reshape_axes_array,
    draw_evoked_response,
    ChannelsHelper,
    render_label_grid,
)

import matplotlib.pyplot as plt
import numpy as np
import mne
import copy
from tqdm.auto import tqdm
import logging

plt.ioff()

plot_registry = {}


def register_plot(name, dto_cls):
    def decorator(func):
        plot_registry[name] = {
            "params": dto_cls,
            "function": func
        }
        return func

    return decorator


def prepare_channels(inst, params):
    """Pick channels by name, optionally filter by complete-trace µV range, then optionally combine."""
    try:
        if hasattr(inst, "event_id") and hasattr(inst, "load_data"):
            inst = inst.copy().load_data()
    except Exception:
        pass

    ch_helper = ChannelsHelper(params, inst)
    ch_helper.pick_channels()
    ch_helper.filter_by_uv()
    picks = ch_helper.picks or []
    pick_names = ch_helper.pick_names or []

    if getattr(params, 'combine_channels', False):
        if pick_names:
            inst = mne.channels.combine_channels(
                inst, groups={"combined": list(pick_names)}, method="mean"
            )
        else:
            inst = inst.copy()
    else:
        inst = inst.copy().pick(picks)

    return inst


class EEGVisualization:
    def __init__(self, get_raw_func, get_epochs_func, get_task_func, get_evoked_func):
        self.get_raw = get_raw_func
        self.get_epochs = get_epochs_func
        self.get_task = get_task_func
        self.get_evoked = get_evoked_func
        # Setup class logger using provided name variable
        name = __name__
        self._log = logging.getLogger(name)
        self.spec = plot_registry
        for key in self.spec:
            func = self.spec[key]["function"]
            self.spec[key]["function"] = func.__get__(self)

    # ----- shared helpers for grid plots -----
    def _snr_spectrum(self, psd: np.ndarray, noise_n_neighbor_freqs: int = 3, noise_skip_neighbor_freqs: int = 1) -> np.ndarray:
        """Compute SNR by dividing PSD by a neighborhood-averaged noise estimate along the last axis.

        psd shape: (..., n_freqs)
        returns SNR with the same shape.
        """
        kernel = np.concatenate((
            np.ones(noise_n_neighbor_freqs),
            np.zeros(2 * noise_skip_neighbor_freqs + 1),
            np.ones(noise_n_neighbor_freqs),
        ))
        kernel /= kernel.sum()
        mean_noise = np.apply_along_axis(
            lambda psd_: np.convolve(psd_, kernel, mode="valid"),
            axis=-1,
            arr=psd,
        )
        edge_width = noise_n_neighbor_freqs + noise_skip_neighbor_freqs
        pad = [(0, 0)] * (mean_noise.ndim - 1) + [(edge_width, edge_width)]
        mean_noise = np.pad(mean_noise, pad_width=tuple(pad), constant_values=float('nan'))
        return psd / mean_noise

    def prepare_params(self, task_dto, key):
        spec = self.spec[key]
        params_cls = spec["params"]
        if not params_cls:
            return {}

        params_obj = params_cls()

        if EpochParamsDTO and isinstance(params_obj, EpochParamsDTO):
            _epochs, labels = self.get_epochs(task_dto=task_dto, epoch_params=params_obj)

            if isinstance(labels, str) and labels == "unavailable":
                return {}
            if labels is not None:
                return {"stimulus": [None] + list(labels)}

        return {}

    @register_plot("Sensor Layout", FilterParamsDTO)
    def plot_sensors(self, task_dto: BaseTaskDTO, params: FilterParamsDTO):
        raw = self.get_raw(task_dto, params)
        raw.pick(params.channels_list)
        fig = raw.plot_sensors(show_names=True)
        return [fig]

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

    @register_plot("PSD Grid", EpochPSDParamsDTO)
    def plot_psd_grid(self, task_dto: BaseTaskDTO, params: EpochPSDParamsDTO):
        """Render a grid of PSDs organized by label tokens (refactored to shared grid)."""
        epochs, available_labels = self.get_epochs(task_dto, params)
        if epochs is None:
            return None
        # Welch parameters
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
        epochs, labels = self.get_epochs(task_dto, params)
        if epochs is None:
            return None
        fig_list = []
        for condition in epochs.event_id:
            copy_params = copy.deepcopy(params)
            copy_params.stimulus = condition
            evk = self.get_evoked(task_dto, copy_params)
            if evk is None:
                continue
            evk = prepare_channels(evk, copy_params)
            fig = evk.plot(gfp=copy_params.gfp, spatial_colors=copy_params.spatial_colors, show=False)
            fig = finalize_figure(fig, task_dto, condition, caption_line=str(copy_params), plot_name="Evoked per Condition")
            fig_list.append(fig)
        return fig_list

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

        def snr_spectrum(psd, noise_n_neighbor_freqs=3, noise_skip_neighbor_freqs=1):
            averaging_kernel = np.concatenate((
                np.ones(noise_n_neighbor_freqs),
                np.zeros(2 * noise_skip_neighbor_freqs + 1),
                np.ones(noise_n_neighbor_freqs),
            ))
            averaging_kernel /= averaging_kernel.sum()
            mean_noise = np.apply_along_axis(
                lambda psd_: np.convolve(psd_, averaging_kernel, mode="valid"),
                axis=-1,
                arr=psd,
            )
            edge_width = noise_n_neighbor_freqs + noise_skip_neighbor_freqs
            pad_width = [(0, 0)] * (mean_noise.ndim - 1) + [(edge_width, edge_width)]
            mean_noise = np.pad(mean_noise, pad_width=tuple(pad_width), constant_values=float('nan'))
            return psd / mean_noise

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

    @register_plot("SNR Grid", EpochPSDParamsDTO)
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

    @register_plot("Evoked Grid", EvokedParamsDTO)
    def plot_evoked_grid(self, task_dto: BaseTaskDTO, params: EvokedParamsDTO):
        """Render a grid of evoked responses organized by label tokens, via shared grid helper."""
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
