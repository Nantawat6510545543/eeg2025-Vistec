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
from tqdm.auto import tqdm

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
    """Pick channels by name, optionally filter by complete-trace µV range, then optionally combine.

    Behavior of µV range filter:
    - If params.uv_min is not None, a channel is kept only if its minimum over all samples >= uv_min.
    - If params.uv_max is not None, a channel is kept only if its maximum over all samples <= uv_max.
    The data are converted from Volts to microvolts before comparison.
    """
    ch = getattr(params, 'channels_list', []) or []
    picks_raw = mne.pick_channels(inst.ch_names, include=ch)
    try:
        picks = [int(i) for i in np.array(picks_raw).tolist()]
    except Exception:
        picks = list(picks_raw) if isinstance(picks_raw, (list, tuple)) else []
    pick_names = [inst.ch_names[i] for i in picks]

    # Coerce uv_min/uv_max to floats or None (UI may provide empty strings)
    def _to_float_or_none(x):
        if x is None:
            return None
        try:
            # treat blank strings as None
            if isinstance(x, str) and x.strip() == "":
                return None
            return float(x)
        except Exception:
            return None

    uv_min = _to_float_or_none(getattr(params, 'uv_min', None))
    uv_max = _to_float_or_none(getattr(params, 'uv_max', None))

    def _filter_picks_by_uv(inst_obj, pick_idx_list):
        if pick_idx_list is None or pick_idx_list.size == 0:
            return []
        # Determine data array in µV for the selected picks
        data_uv = None
        try:
            # Evoked has .data (n_channels, n_times)
            if hasattr(inst_obj, 'data') and not hasattr(inst_obj, 'get_data'):
                data_uv = inst_obj.data[pick_idx_list, :] * 1e6
            else:
                # Raw and Epochs both have get_data; shapes differ:
                # Raw: (n_channels, n_times), Epochs: (n_epochs, n_channels, n_times)
                arr = inst_obj.get_data(picks=pick_idx_list)
                if arr.ndim == 2:
                    data_uv = arr * 1e6  # (n_ch, n_times)
                elif arr.ndim == 3:
                    # reduce over epochs: take global min/max over all epochs/time
                    # reshape to (n_ch, -1) for uniform handling
                    n_epochs, n_ch, n_times = arr.shape
                    data_uv = arr.transpose(1, 0, 2).reshape(n_ch, -1) * 1e6
                else:
                    # Unexpected shape; skip filtering
                    return pick_idx_list
        except Exception:
            # If data access fails (e.g., not preloaded), fall back to no additional filtering
            return pick_idx_list

        if data_uv is None:
            return pick_idx_list

        ch_mins = np.nanmin(data_uv, axis=-1)
        ch_maxs = np.nanmax(data_uv, axis=-1)
        keep_mask = np.ones(len(pick_idx_list), dtype=bool)
        if uv_min is not None:
            keep_mask &= ch_mins >= uv_min
        if uv_max is not None:
            keep_mask &= ch_maxs <= uv_max

        kept = [idx for idx, keep in zip(pick_idx_list, keep_mask) if keep]
        return kept

    # Apply µV filtering if any bound is specified
    if uv_min is not None or uv_max is not None:
        filtered_picks = _filter_picks_by_uv(inst, picks)
        if not filtered_picks:
            # Fallback: if everything filtered out, keep original picks to avoid empty selection
            filtered_picks = picks
        picks = filtered_picks
        pick_names = [inst.ch_names[i] for i in picks]

    # Apply combine or simple pick
    if getattr(params, 'combine_channels', False):
        # Only combine if we have at least one channel
        if pick_names:
            inst = mne.channels.combine_channels(
                inst, groups={"combined": list(pick_names)}, method="mean"
            )
        else:
            # Nothing to combine; leave as-is
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
        return [finalize_figure(fig, task_dto, caption=str(params), plot_name="Time Domain")]

    @register_plot("Frequency Domain", PSDParamsDTO)
    def plot_frequency(self, task_dto: BaseTaskDTO, params: PSDParamsDTO):
        raw = self.get_raw(task_dto, params)
        raw = prepare_channels(raw, params)
        psd = raw.compute_psd(fmin=params.fmin, fmax=params.fmax)
        fig = psd.plot(average=params.average, spatial_colors=params.spatial_colors, dB=params.dB, show=False)
        return [finalize_figure(fig, task_dto, caption=str(params), plot_name="Frequency Domain")]

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
            fig = finalize_figure(fig, task_dto, condition, caption=str(params), plot_name="Condition-wise PSD")
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
        return finalize_figure(fig, task_dto, params.stimulus, caption=str(params), plot_name="Epoch Plot")

    @register_plot("Evoked Plot", EvokedParamsDTO)
    def plot_evoked(self, task_dto: BaseTaskDTO, params: EvokedParamsDTO):
        evoked = self.get_evoked(task_dto, params)
        if evoked is None:
            return None
        evoked = prepare_channels(evoked, params)
        fig = evoked.plot(gfp=params.gfp, spatial_colors=params.spatial_colors, show=False)
        return finalize_figure(fig, task_dto, params.stimulus, caption=str(params), plot_name="Evoked Plot")

    @register_plot("Evoked Topo Plot", EvokedTopoParamsDTO)
    def plot_evoked_topo(self, task_dto: BaseTaskDTO, params: EvokedTopoParamsDTO):
        params.combine_channels = False
        evoked = self.get_evoked(task_dto, params)
        if evoked is None:
            return None
        evoked = prepare_channels(evoked, params)
        fig = evoked.plot_topomap(times=params.get_times, show=False)
        return finalize_figure(fig, task_dto, params.stimulus, caption=str(params), plot_name="Evoked Topo")

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
        return finalize_figure(fig, task_dto, params.stimulus, caption=str(params), plot_name="Evoked Joint")

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
        return finalize_figure(fig, task_dto, params.stimulus, caption=str(params), plot_name="SNR Spectrum")

    # complete v5
    @register_plot("Evoked Grid", EvokedParamsDTO)
    def plot_evoked_grid(self, task_dto: BaseTaskDTO, params: EvokedParamsDTO):
        """Render a grid of evoked responses organized by label tokens, with two scale modes.

        Behavior:
        - Labels are split by '_' into tokens; the grid uses up to 3 tokens as (page, column, row).
        - Titles (top row) are always drawn.
        - Cells with no corresponding label or evoked data are left as empty plots with visible axes.
        - The bottom row always shows the time axis labels. All cells share x-limits [tmin, tmax].
        - Each subplot shows a small "µV" unit tag at its top-left.
        - Each non-empty subplot shows the epoch count used in averaging at top-right as "n=..".

        scale_mode:
        - "per-plot" (default): each cell autoscales; all cells show their own y-tick numbers (no row text labels).
        - "uniform-grid": compute a global symmetric y-range (µV) per page and apply to all cells; inner columns hide y-tick labels.
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
                axis.set_ylabel(f"{row_token}")
                axis.tick_params(labelleft=True)
            else:
                # Per user request: in per-plot mode show numeric ticks on all subplots; otherwise hide
                if scale_mode == 'per-plot':
                    axis.tick_params(labelleft=True)
                else:
                    axis.tick_params(labelleft=False)

            axis.text(0.01, 1, "µV", transform=axis.transAxes, ha='left', va='bottom', fontsize=8, color='0.4')

        # --- build figures --------------------------------------------------
        total_cells = len(page_values) * num_rows * num_cols
        with tqdm(total=total_cells, desc="Evoked Grid cells", leave=False) as pbar:
            for page_token in page_values:
                fig, axes = plt.subplots(num_rows, num_cols, sharex=False, sharey=False)
                axes_2d = reshape_axes_array(axes, num_rows, num_cols)

                # Track global y-range across this page for uniform-grid mode
                y_min, y_max = None, None

                for row_idx, row_token in enumerate(row_values):
                    for col_idx, col_token in enumerate(column_values):
                        axis = axes_2d[row_idx, col_idx]
                        axis.cla()

                        label = cell_to_label_map.get((page_token, col_token, row_token))
                        if label is not None:
                            evoked, effective_params = _fetch_evoked_response(label)
                            if evoked is not None:
                                data_uv = evoked.data * 1e6
                                if data_uv.size:
                                    dmin = float(np.nanmin(data_uv))
                                    dmax = float(np.nanmax(data_uv))
                                    if np.isfinite(dmin) and np.isfinite(dmax):
                                        y_min = dmin if y_min is None else min(y_min, dmin)
                                        y_max = dmax if y_max is None else max(y_max, dmax)

                            draw_evoked_response(axis, evoked, effective_params)
                            # Annotate epoch count (nave) on top-right
                            try:
                                nave = getattr(evoked, 'nave', None)
                                if nave is not None:
                                    axis.text(1, 1, f"n={int(nave)}", transform=axis.transAxes,
                                              ha='right', va='bottom', fontsize=8, color='0.4')
                            except Exception:
                                pass

                        # set labels and axis limits for every cell, even if empty
                        _label_cell(axis, row_idx, col_idx, row_token, col_token)
                        axis.set_xlim(params.tmin, params.tmax)

                        pbar.update(1)

                # Harmonize y-limits across all subplots in this page when requested
                if scale_mode == 'uniform-grid' and y_min is not None and y_max is not None:
                    # symmetric around 0 using max absolute amplitude
                    y_abs = float(max(abs(y_min), abs(y_max)))
                    if not (y_abs and np.isfinite(y_abs) and y_abs > 0):
                        y_abs = 1.0
                    y_lo, y_hi = -y_abs, y_abs
                    for r in range(num_rows):
                        for c in range(num_cols):
                            axes_2d[r, c].set_ylim(y_lo, y_hi)

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

                # finalize and caption
                page_stimulus = page_token if (grid_mode == 3 and page_token is not None) else None
                fig = finalize_figure(
                    fig,
                    task_dto,
                    stimulus=page_stimulus,
                    caption_line=str(params),
                    plot_name="Evoked Grid",
                )
                figures.append(fig)

        return figures
