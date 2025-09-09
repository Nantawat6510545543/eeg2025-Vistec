from ..models import (
    BaseTaskDTO, FilterParamsDTO, TimeDomainParamsDTO, PSDParamsDTO,
    EpochParamsDTO, EpochPSDParamsDTO
)
from ..utils import finalize_figure
import matplotlib.pyplot as plt

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
        raw.plot_sensors(show_names=True)

    @register_plot("Time Domain Plot", TimeDomainParamsDTO)
    def plot_time(self, task_dto: BaseTaskDTO, params: TimeDomainParamsDTO):
        raw = self.get_raw(task_dto, params)
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
        psd = raw.compute_psd(fmin=params.fmin, fmax=params.fmax)
        fig = psd.plot(average=params.average, spatial_colors=params.spatial_colors, dB=params.dB, show=False)
        return [finalize_figure(fig, task_dto, caption=vars(params), plot_name="Frequency Domain")]

    @register_plot("Condition-wise PSD", EpochPSDParamsDTO)
    def plot_conditionwise_psd(self, task_dto: BaseTaskDTO, params: EpochPSDParamsDTO):
        epochs, labels = self.get_epochs(task_dto, params)
        if epochs is None:
            print(f"No epochs available for {task_dto.subject} - {task_dto.task}")
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
        return self._plot_epochs_base(task_dto, params, mode="Epochs")

    @register_plot("Evoked Response", EpochParamsDTO)
    def plot_evoked(self, task_dto: BaseTaskDTO, params: EpochParamsDTO):
        return self._plot_epochs_base(task_dto, params, mode="Evoked")

    def _plot_epochs_base(self, task_dto, params, mode):
        epochs, labels = self.get_epochs(task_dto, params)
        if epochs is None:
            return None
        if params.stimulus and params.stimulus in epochs.event_id:
            epochs = epochs[params.stimulus]

        if mode == "Epochs":
            fig = epochs.plot(events=False, n_channels=params.n_channels, show=False)
        elif mode == "Evoked":
            fig = epochs.average().plot_joint(show=False)

        return [finalize_figure(fig, task_dto, params.stimulus, caption=vars(params), plot_name=mode)]
