from ..models import *
from ..utils import  (finalize_figure,
    split_tokens,
    compute_axes_values,
    map_cells_to_labels,
    reshape_axes_array,
    draw_evoked_response,
)

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

    # complete v4
    @register_plot("Evoked Grid", EvokedParamsDTO)
    def plot_evoked_grid(self, task_dto: BaseTaskDTO, params: EvokedParamsDTO):
        """Render a grid of evoked responses organized by label tokens, with two scale modes.

        Behavior:
        - Labels are split by '_' into tokens; the grid uses up to 3 tokens as (page, column, row).
        - Titles (top row) and y-labels (first column) are always drawn.
        - Cells with no corresponding label or evoked data are left as empty plots with visible axes.
        - The bottom row always shows the time axis labels. All cells share x-limits [tmin, tmax].
        - All subplots within a page share the same y-limits (computed from available data, in µV).

        scale_mode:
        - "per-plot" (default): each cell autoscale; add a small per-cell y-scale label.
        - "uniform-grid": compute a global y-range (µV) per page and apply to all cells.
        """
        # 1) Discover available labels (from epochs) to infer the grid shape
        epochs, available_labels = self.get_epochs(task_dto, params)
        if epochs is None or not available_labels:
            return None

        # Determine scaling mode (Dropdown returns a single selected string)
        scale_mode = getattr(params, 'scale_mode', 'per-plot')
        if isinstance(scale_mode, (list, tuple)) and scale_mode:
            scale_mode = scale_mode[0]

        # --- tokenization ----------------------------------------------------
        tokens_by_label = {label: split_tokens(label) for label in available_labels}
        max_token_count = max((len(tokens) for tokens in tokens_by_label.values()), default=1)
        grid_mode = min(max_token_count, 3)  # clamp to 3 max

        # --- grid axes + label mapping --------------------------------------
        page_values, column_values, row_values = compute_axes_values(tokens_by_label, grid_mode)
        cell_to_label_map = map_cells_to_labels(tokens_by_label, grid_mode)

        figures = []
        num_rows, num_cols = len(row_values), len(column_values)

        # --- fetch evoked helper --------------------------------------------
        def _fetch_evoked_response(label: str):
            params_copy = copy.deepcopy(params)
            params_copy.stimulus = [label]
            evoked = self.get_evoked(task_dto, params_copy)
            return (prepare_channels(evoked, params_copy), params_copy) if evoked is not None else (None, params_copy)

        # --- label cell helper ----------------------------------------------
        def _label_cell(axis, row_idx, col_idx, row_token, col_token):
            if row_idx == 0 and (col_token is not None and col_token != ""):
                axis.set_title(col_token)
            if col_idx == 0:
                ylabel = f"{row_token} (\u00b5V)" if (row_token is not None and row_token != "") else "\u00b5V"
                axis.set_ylabel(ylabel)
                axis.tick_params(labelleft=True)
            else:
                # Per user request: in per-plot mode show numeric ticks on all subplots; otherwise hide
                if scale_mode == 'per-plot':
                    axis.tick_params(labelleft=True)
                else:
                    axis.tick_params(labelleft=False)

        # --- build figures --------------------------------------------------
        for page_token in page_values:
            fig, axes = plt.subplots(num_rows, num_cols, sharex=False, sharey=False)
            axes_2d = reshape_axes_array(axes, num_rows, num_cols)

            # Track global y-range across this page for uniform-grid mode
            y_min, y_max = None, None

            for row_idx, row_token in enumerate(row_values):
                for col_idx, col_token in enumerate(column_values):
                    axis = axes_2d[row_idx, col_idx]
                    try:
                        axis.cla()  # always clear first
                    except Exception:
                        pass

                    label = cell_to_label_map.get((page_token, col_token, row_token))
                    did_plot = False
                    if label is not None:
                        evoked, effective_params = _fetch_evoked_response(label)
                        if evoked is not None:
                            # Update global y-range from channel data in µV for uniform-grid
                            try:
                                data_uv = evoked.data * 1e6
                                if data_uv.size:
                                    dmin = float(np.nanmin(data_uv))
                                    dmax = float(np.nanmax(data_uv))
                                    if np.isfinite(dmin) and np.isfinite(dmax):
                                        y_min = dmin if y_min is None else min(y_min, dmin)
                                        y_max = dmax if y_max is None else max(y_max, dmax)
                            except Exception:
                                pass
                            # Draw actual evoked traces
                            draw_evoked_response(axis, evoked, effective_params)
                            did_plot = True

                    # set labels and axis limits
                    _label_cell(axis, row_idx, col_idx, row_token, col_token)
                    try:
                        axis.set_xlim(params.tmin, params.tmax)
                    except Exception:
                        pass


            # Harmonize y-limits across all subplots in this page when requested
            if scale_mode == 'uniform-grid' and y_min is not None and y_max is not None:
                if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min == y_max:
                    pad = 1.0
                    y_min, y_max = (y_min or -pad) - pad, (y_max or pad) + pad
                for r in range(num_rows):
                    for c in range(num_cols):
                        try:
                            axes_2d[r, c].set_ylim(y_min, y_max)
                        except Exception:
                            pass

            # x-ticks only on bottom row
            last_row_idx = num_rows - 1
            for r in range(num_rows):
                for c in range(num_cols):
                    axis = axes_2d[r, c]
                    if r == last_row_idx:
                        axis.set_xlabel("Time [s]")
                        axis.tick_params(labelbottom=True)
                    else:
                        axis.tick_params(labelbottom=False)

            page_stimulus = page_token if (grid_mode == 3 and page_token is not None) else None
            fig = finalize_figure(
                fig,
                task_dto,
                stimulus=page_stimulus,
                caption=vars(params),
                plot_name="Evoked Grid",
            )
            figures.append(fig)

        return figures
