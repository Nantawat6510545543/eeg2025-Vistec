from ..models import *
from ..utils import finalize_figure
import matplotlib.pyplot as plt
import numpy as np

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


class EEGVisualization:
    def __init__(self, get_raw_func, get_epochs_func, get_task_func):
        self.get_raw = get_raw_func
        self.get_epochs = get_epochs_func
        self.get_task = get_task_func
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
        raw = raw.pick(params.channels_list)
        fig = raw.plot_sensors(show_names=True)
        return [fig]

    @register_plot("Time Domain Plot", TimeDomainParamsDTO)
    def plot_time(self, task_dto: BaseTaskDTO, params: TimeDomainParamsDTO):
        raw = self.get_raw(task_dto, params)
        raw = raw.pick(params.channels_list)
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
        raw = raw.pick(params.channels_list)
        psd = raw.compute_psd(fmin=params.fmin, fmax=params.fmax)
        fig = psd.plot(average=params.average, spatial_colors=params.spatial_colors, dB=params.dB, show=False)
        return [finalize_figure(fig, task_dto, caption=vars(params), plot_name="Frequency Domain")]

    @register_plot("Condition-wise PSD", EpochPSDParamsDTO)
    def plot_conditionwise_psd(self, task_dto: BaseTaskDTO, params: EpochPSDParamsDTO):
        epochs, labels = self.get_epochs(task_dto, params)
        epochs = epochs.pick(params.channels_list)
        if epochs is None:
            return None
        fig_list = []
        for condition in epochs.event_id:
            condition_epochs = epochs[condition]
            if len(condition_epochs) == 0:
                continue
            psd = condition_epochs.compute_psd(fmin=params.fmin, fmax=params.fmax)
            fig = psd.plot(average=params.average, spatial_colors=params.spatial_colors, dB=params.dB, show=False)
            fig = finalize_figure(fig, task_dto, condition, caption=vars(params), plot_name="Condition-wise PSD")
            fig_list.append(fig)
        return fig_list

    @register_plot("Epoch Plot", EpochParamsDTO)
    def plot_epochs(self, task_dto: BaseTaskDTO, params: EpochParamsDTO):
        epochs, labels = self.get_epochs(task_dto, params)
        epochs = epochs.pick(params.channels_list)
        if epochs is None:
            return None
        if params.stimulus and params.stimulus in epochs.event_id:
            epochs = epochs[params.stimulus]

        fig = epochs.plot(events=False, n_channels=params.n_channels, show=False)

        return [finalize_figure(fig, task_dto, params.stimulus, caption=vars(params), plot_name="Epoch Plot")]

    @register_plot("Evoked Plot", EvokedParamsDTO)
    def plot_evoked(self, task_dto: BaseTaskDTO, params: EvokedParamsDTO):
        epochs, labels = self.get_epochs(task_dto, params)
        if epochs is None:
            return None

        epochs = epochs.pick(params.channels_list)
        if params.stimulus and params.stimulus in epochs.event_id:
            epochs = epochs[params.stimulus]

        evoked = epochs.average()
        fig = evoked.plot(gfp=params.gfp, spatial_colors=params.spatial_colors, show=False)

        return [finalize_figure(fig, task_dto, params.stimulus, caption=vars(params), plot_name="Evoked Plot")]

    @register_plot("Evoked Topo Plot", EvokedTopoParamsDTO)
    def plot_evoked_topo(self, task_dto: BaseTaskDTO, params: EvokedTopoParamsDTO):
        epochs, labels = self.get_epochs(task_dto, params)
        if epochs is None:
            return None

        epochs = epochs.pick(params.channels_list)
        if params.stimulus and params.stimulus in epochs.event_id:
            epochs = epochs[params.stimulus]

        evoked = epochs.average()
        fig = evoked.plot_topomap(times=params.times, show=False)

        return [finalize_figure(fig, task_dto, params.stimulus, caption=vars(params), plot_name="Evoked Topo")]

    @register_plot("Evoked Plot Joint", EvokedJointParamsDTO)
    def plot_evoked_joint(self, task_dto: BaseTaskDTO, params: EvokedJointParamsDTO):
        epochs, labels = self.get_epochs(task_dto, params)
        if epochs is None:
            return None

        epochs = epochs.pick(params.channels_list)
        if params.stimulus and params.stimulus in epochs.event_id:
            epochs = epochs[params.stimulus]

        evoked = epochs.average()
        fig = evoked.plot_joint(
            times=params.times,
            topomap_args={},
            ts_args={"gfp": params.gfp, "spatial_colors": params.spatial_colors},
            show=False,
        )

        return [finalize_figure(fig, task_dto, params.stimulus, caption=vars(params), plot_name="Evoked Joint")]

    @register_plot("SNR Spectrum", EpochPSDParamsDTO)
    def plot_snr(self, task_dto: BaseTaskDTO, params: EpochPSDParamsDTO):
        epochs, labels = self.get_epochs(task_dto, params)
        epochs = epochs.pick(params.channels_list)

        sfreq = epochs.info["sfreq"]

        # --- Compute PSD from epochs ---
        spectrum = epochs.compute_psd(
            method="welch",
            n_fft=int(max(8, sfreq * (params.tmax - params.tmin))),  # keep a minimal n_fft
            n_overlap=0,
            n_per_seg=None,
            tmin=params.tmin,
            tmax=params.tmax,
            fmin=params.fmin,
            fmax=params.fmax,
            window="boxcar",
            verbose=False,
        )
        psds, freqs = spectrum.get_data(return_freqs=True)  # (n_epochs, n_channels, n_freqs)

        # --- SNR calculation ---
        def snr_spectrum(psd, noise_n_neighbor_freqs=3, noise_skip_neighbor_freqs=1):
            averaging_kernel = np.concatenate(
                (
                    np.ones(noise_n_neighbor_freqs),
                    np.zeros(2 * noise_skip_neighbor_freqs + 1),
                    np.ones(noise_n_neighbor_freqs),
                )
            )
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

        # --- Plotting ---
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

        # PSD spectrum (dB)
        psds_db = 10 * np.log10(psds, where=psds > 0, out=np.full_like(psds, np.nan))
        psds_mean = np.nanmean(psds_db[..., freq_idx], axis=(0, 1))
        psds_std = np.nanstd(psds_db[..., freq_idx], axis=(0, 1))
        axes[0].plot(freqs[freq_idx], psds_mean, color="b")
        axes[0].fill_between(
            freqs[freq_idx], psds_mean - psds_std, psds_mean + psds_std, color="b", alpha=0.2
        )
        axes[0].set(title="PSD spectrum", ylabel="Power Spectral Density [dB]")

        # SNR spectrum
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

        return [finalize_figure(fig, task_dto, params.stimulus, caption=vars(params), plot_name="SNR Spectrum")]
