from ..models import *
from ..utils import finalize_figure
import matplotlib.pyplot as plt
import numpy as np
import mne
import copy

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
    ch = params.channels_list
    picks = mne.pick_channels(inst.ch_names, include=ch)
    if params.combine_channels:
        inst = mne.channels.combine_channels(
            inst, groups={"combined": list(picks)}, method="mean"
        )
    else:
        inst = inst.copy().pick(picks)

    return inst


class EEGVisualization:
    def __init__(self, get_raw_func, get_epochs_func, get_task_func, get_evoked_func):
        self.get_raw = get_raw_func
        self.get_epochs = get_epochs_func
        self.get_task = get_task_func
        self.get_evoked = get_evoked_func
        self.spec = plot_registry
        for key in self.spec:
            func = self.spec[key]["function"]
            self.spec[key]["function"] = func.__get__(self)

    def prepare_params(self, task_dto, key):
        spec = self.spec[key]
        params_cls = spec["params"]
        if not params_cls:
            return {}

        params_obj = params_cls()

        if EpochParamsDTO and isinstance(params_obj, EpochParamsDTO):
            params_obj.only_labels = True
            _epochs, labels = self.get_epochs(task_dto=task_dto, epoch_params=params_obj)
            params_obj.only_labels = False

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
        return [finalize_figure(fig, task_dto, caption=vars(params), plot_name="Time Domain")]

    @register_plot("Frequency Domain", PSDParamsDTO)
    def plot_frequency(self, task_dto: BaseTaskDTO, params: PSDParamsDTO):
        raw = self.get_raw(task_dto, params)
        raw = prepare_channels(raw, params)
        psd = raw.compute_psd(fmin=params.fmin, fmax=params.fmax)
        fig = psd.plot(average=params.average, spatial_colors=params.spatial_colors, dB=params.dB, show=False)
        return [finalize_figure(fig, task_dto, caption=vars(params), plot_name="Frequency Domain")]

    @register_plot("Condition-wise PSD", EpochPSDParamsDTO)
    def plot_conditionwise_psd(self, task_dto: BaseTaskDTO, params: EpochPSDParamsDTO):
        epochs, labels = self.get_epochs(task_dto, params)
        if epochs is None:
            return None
        epochs = prepare_channels(epochs, params)
        fig_list = []
        for condition in epochs.event_id:
            condition_epochs = epochs[condition]
            if len(condition_epochs) == 0:
                continue
            psd = condition_epochs.compute_psd(fmin=params.fmin, fmax=params.fmax, average='mean')
            fig = psd.plot(average=params.average, spatial_colors=params.spatial_colors, dB=params.dB, show=False)
            fig = finalize_figure(fig, task_dto, condition, caption=vars(params), plot_name="Condition-wise PSD")
            fig_list.append(fig)
        return fig_list

    @register_plot("Epoch Plot", EpochParamsDTO)
    def plot_epochs(self, task_dto: BaseTaskDTO, params: EpochParamsDTO):
        epochs, labels = self.get_epochs(task_dto, params)
        if epochs is None:
            return None
        if params.stimulus and params.stimulus in epochs.event_id:
            epochs = epochs[params.stimulus]
        epochs = prepare_channels(epochs, params)
        fig = epochs.plot(events=False, show=False)
        return finalize_figure(fig, task_dto, params.stimulus, caption=vars(params), plot_name="Epoch Plot")

    @register_plot("Evoked Plot", EvokedParamsDTO)
    def plot_evoked(self, task_dto: BaseTaskDTO, params: EvokedParamsDTO):
        evoked = self.get_evoked(task_dto, params)
        if evoked is None:
            return None
        evoked = prepare_channels(evoked, params)
        fig = evoked.plot(gfp=params.gfp, spatial_colors=params.spatial_colors, show=False)
        return finalize_figure(fig, task_dto, params.stimulus, caption=vars(params), plot_name="Evoked Plot")

    @register_plot("Evoked Topo Plot", EvokedTopoParamsDTO)
    def plot_evoked_topo(self, task_dto: BaseTaskDTO, params: EvokedTopoParamsDTO):
        params.combine_channels = False
        evoked = self.get_evoked(task_dto, params)
        if evoked is None:
            return None
        evoked = prepare_channels(evoked, params)
        fig = evoked.plot_topomap(times=params.get_times, show=False)
        return finalize_figure(fig, task_dto, params.stimulus, caption=vars(params), plot_name="Evoked Topo")

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
        return finalize_figure(fig, task_dto, params.stimulus, caption=vars(params), plot_name="Evoked Joint")

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
            fig = finalize_figure(fig, task_dto, condition, caption=vars(copy_params), plot_name="Evoked per Condition")
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
            window="boxcar",
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
            mean_noise = np.pad(mean_noise, pad_width=pad_width, constant_values=np.nan)
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
        return finalize_figure(fig, task_dto, params.stimulus, caption=vars(params), plot_name="SNR Spectrum")

    # complete v1
    @register_plot("Evoked Grid", EvokedParamsDTO)
    def plot_evoked_grid(self, task_dto: BaseTaskDTO, params: EvokedParamsDTO):
        # Discover labels from epochs
        epochs, labels = self.get_epochs(task_dto, params)
        if epochs is None:
            return None

        # Decide grid dimensionality based on max number of tokens across labels
        def split_tokens(lbl: str):
            return [t for t in (lbl or '').split('_') if t != '']

        tokens_by_label = {lbl: split_tokens(lbl) for lbl in labels}
        max_parts = max(len(toks) for toks in tokens_by_label.values())
        mode = 3 if max_parts >= 3 else (2 if max_parts == 2 else 1)

        # Build unique axes for pages/cols/rows
        if mode == 1:
            pages_vals = [None]
            cols_vals = [None]
            rows_vals = sorted({toks[0] for toks in tokens_by_label.values() if len(toks) >= 1})
        elif mode == 2:
            pages_vals = [None]
            cols_vals = sorted({toks[0] for toks in tokens_by_label.values() if len(toks) >= 1})
            rows_vals = sorted({toks[1] for toks in tokens_by_label.values() if len(toks) >= 2})
        else:  # mode >= 3
            pages_vals = sorted({toks[0] for toks in tokens_by_label.values() if len(toks) >= 1})
            cols_vals = sorted({toks[1] for toks in tokens_by_label.values() if len(toks) >= 2})
            rows_vals = sorted({toks[2] for toks in tokens_by_label.values() if len(toks) >= 3})

        # Map (page, col, row) -> exact label string if available
        cell_to_label = {}
        for lbl, toks in tokens_by_label.items():
            if mode == 1 and len(toks) >= 1:
                key = (None, None, toks[0])
            elif mode == 2 and len(toks) >= 2:
                key = (None, toks[0], toks[1])
            elif mode == 3 and len(toks) >= 3:
                key = (toks[0], toks[1], toks[2])
            else:
                continue
            # first-come keeps mapping; ensures determinism
            cell_to_label.setdefault(key, lbl)

        figures = []
        n_rows, n_cols = len(rows_vals), len(cols_vals)

        # Helper to normalize axes array
        def to_2d_axes(axs, R, C):
            if R == 1 and C == 1:
                return np.array([[axs]])
            if R == 1:
                return np.array([axs])
            if C == 1:
                return np.array([[a] for a in axs])
            return axs

        # Iterate pages (or single None page)
        for page in pages_vals:
            fig, axs = plt.subplots(n_rows, n_cols, sharex=False, sharey=False)
            axs2d = to_2d_axes(axs, n_rows, n_cols)

            for r_idx, r_token in enumerate(rows_vals):
                for c_idx, c_token in enumerate(cols_vals):
                    key = (page, c_token, r_token)
                    lbl = cell_to_label.get(key)
                    ax = axs2d[r_idx, c_idx]

                    # Always start from a clean axes so previous content doesn't leak
                    try:
                        ax.cla()
                    except Exception:
                        pass

                    # Plot data if available
                    has_content = False
                    if lbl is not None:
                        p = copy.deepcopy(params)
                        p.stimulus = lbl
                        evk = self.get_evoked(task_dto, p)
                        if evk is not None:
                            evk = prepare_channels(evk, p)
                            times = evk.times
                            data_uv = evk.data * 1e6  # (n_channels, n_times)

                            gfp_mode = p.gfp
                            if gfp_mode != "only":
                                for ch in range(data_uv.shape[0]):
                                    ax.plot(times, data_uv[ch], color='0.75', linewidth=0.6, zorder=1)

                            if gfp_mode is True or gfp_mode == "only":
                                if data_uv.shape[0] > 1:
                                    gfp_signal = np.std(data_uv, axis=0)
                                else:
                                    gfp_signal = data_uv[0]
                                ax.plot(times, gfp_signal, color='k', linewidth=1.2, zorder=2)

                            # Reference lines for non-empty content
                            ax.axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.6)
                            ax.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.6)
                            has_content = True

                    # Apply titles/labels after clearing/plotting so they persist
                    if r_idx == 0 and (c_token is not None and c_token != ""):
                        ax.set_title(c_token)
                    if c_idx == 0:
                        ylabel = f"{r_token} (\u00b5V)" if (r_token is not None and r_token != "") else "\u00b5V"
                        ax.set_ylabel(ylabel)
                    else:
                        ax.tick_params(labelleft=False)

                    # Keep consistent time axis for both empty and non-empty cells
                    try:
                        ax.set_xlim(params.tmin, params.tmax)
                    except Exception:
                        pass

            # Always treat the last row as the bottom for x labels
            bottom_row_idx = n_rows - 1
            for r_idx in range(n_rows):
                for c_idx in range(n_cols):
                    ax = axs2d[r_idx, c_idx]
                    if r_idx == bottom_row_idx:
                        ax.set_xlabel("Time [s]")
                        ax.tick_params(labelbottom=True)
                    else:
                        ax.tick_params(labelbottom=False)

            # Finalize per-page figure with header and caption
            page_stimulus = page if (mode == 3 and page is not None) else None
            fig = finalize_figure(fig, task_dto, stimulus=page_stimulus, caption=vars(params), plot_name="Evoked Grid")
            figures.append(fig)

        return figures
