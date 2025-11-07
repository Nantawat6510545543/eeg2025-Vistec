from ..models import *

from ..utils import  (finalize_figure,
    prepare_channels,
)

import matplotlib.pyplot as plt
import numpy as np
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


from .base_service import BaseService


class EEGVisualization(BaseService):
    description = "Produces one concise figure for the current EEG selection."
    def __init__(self, get_raw_func, get_epochs_func, get_task_func, get_evoked_func):
        super().__init__(
            registry=plot_registry,
            get_raw_func=get_raw_func,
            get_epochs_func=get_epochs_func,
            get_evoked_func=get_evoked_func,
            get_task_func=get_task_func,
        )

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
            fig = finalize_figure(fig, task_dto, condition, caption_line=str(copy_params),
                                  plot_name="Evoked per Condition")
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
        snrs = self._snr_spectrum(psds)
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