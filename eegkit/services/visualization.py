import pandas as pd
from ..models import (
    TaskDTO, FilterParamsDTO, TimeDomainParamsDTO, PSDParamsDTO,
    EpochParamsDTO, EpochFullParamsDTO, TableInfoDTO, EpochPSDParamsDTO
)
from ..utils import finalize_figure
from copy import deepcopy
import matplotlib.pyplot as plt
plt.ioff()

plot_registry = {"Plot": {}, "Data": {}}
def register_plot(group, name, dto_cls):
    def decorator(func):
        plot_registry[group][name] = {
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
        self.specs = plot_registry

        for group in self.specs:
            for key in self.specs[group]:
                func = self.specs[group][key]["function"]
                self.specs[group][key]["function"] = func.__get__(self)

    def prepare_params(self, task_dto: TaskDTO, group: str, key: str):
        spec = self.specs[group][key]
        params = spec["params"]
        if params is None:
            return 
        updates = {}
        # print(f"preparing with {params} as type {type(params)}")
        if isinstance(params(), EpochParamsDTO):
            epochs, labels = self.get_epochs(task_dto, params)
            # print(labels)
            if labels is not None:
                updates["stimulus"] = [None] + sorted(set(labels))
            else:
                updates["stimulus"] = [None] 

        # print("returning")
        return updates

    @register_plot("Plot", "Sensor Layout", FilterParamsDTO)
    def plot_sensors(self, task_dto: TaskDTO, params: FilterParamsDTO):
        raw = self.get_raw(task_dto, params)
        raw.plot_sensors(show_names=True)

    @register_plot("Plot", "Time Domain Plot", TimeDomainParamsDTO)
    def plot_time(self, task_dto: TaskDTO, params: TimeDomainParamsDTO):
        raw = self.get_raw(task_dto, params)
        fig = raw.plot(
            duration=params.duration,
            start=params.start,
            n_channels=params.n_channels,
            scalings='auto',
            show=False,
        )
        return [finalize_figure(fig, task_dto, caption=vars(params), plot_name="Time Domain")]

    @register_plot("Plot", "Frequency Domain", PSDParamsDTO)
    def plot_frequency(self, task_dto: TaskDTO, params: PSDParamsDTO):
        raw = self.get_raw(task_dto, params)
        psd = raw.compute_psd(fmin=params.fmin, fmax=params.fmax)
        fig = psd.plot(average=params.average, spatial_colors=params.spatial_colors, dB=params.dB, show=False)
        return [finalize_figure(fig, task_dto, caption=vars(params), plot_name="Frequency Domain")]

    @register_plot("Plot", "Condition-wise PSD", EpochPSDParamsDTO)
    def plot_conditionwise_psd(self, task_dto: TaskDTO, params: EpochPSDParamsDTO):
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

    @register_plot("Plot", "Epoch Plot", EpochFullParamsDTO)
    def plot_epochs(self, task_dto: TaskDTO, params: EpochParamsDTO):
        return self._plot_epochs_base(task_dto, params, mode="Epochs")

    @register_plot("Plot", "Evoked Response", EpochParamsDTO)
    def plot_evoked(self, task_dto: TaskDTO, params: EpochParamsDTO):
        return self._plot_epochs_base(task_dto, params, mode="Evoked")

    def _plot_epochs_base(self, task_dto, params, mode):
        epochs, labels = self.get_epochs(task_dto, params)
        if epochs is None:
            print(f"No epochs available for {task_dto.subject} - {task_dto.task}")
            return None
        if params.stimulus and params.stimulus in epochs.event_id:
            epochs = epochs[params.stimulus]

        if mode == "Epochs":
            fig = epochs.plot(events=False, n_channels=params.n_channels, show=False)
        elif mode == "Evoked":
            fig = epochs.average().plot_joint(show=False)

        return [finalize_figure(fig, task_dto, params.stimulus, caption=vars(params), plot_name=mode)]

    @register_plot("Data", "EEG Table", TableInfoDTO)
    def show_table(self, task_dto: TaskDTO, table_info: TableInfoDTO):
        task_model = self.get_task(task_dto)
        df_map = {
            'events': task_model.events,
            'channels': task_model.channels,
            'electrodes': task_model.electrodes
        }
        return df_map.get(table_info.table_type, pd.DataFrame()).head(table_info.rows)

    @register_plot("Data", "Epochs Table", EpochParamsDTO)
    def show_epochs_table(self, task_dto: TaskDTO, params: EpochParamsDTO):
        epochs, labels = self.get_epochs(task_dto, params)
        if epochs is None or labels is None:
            return None

        labels_series = pd.Series(labels)
        unique_labels = sorted(set(labels_series))
        rows = []

        for label in unique_labels:
            # Find indices for this label
            idx = labels_series[labels_series == label].index
            label_epochs = epochs[idx]
            row = {
                'label': label,
                'n_epochs': len(label_epochs),
                'n_channels': len(label_epochs.ch_names),
                'timespan_sec': label_epochs.times[-1] - label_epochs.times[0] if len(label_epochs.times) > 1 else 0,
                'sampling_rate': label_epochs.info['sfreq'],
                'duration_per_epoch_sec': label_epochs.get_data().shape[-1] / label_epochs.info['sfreq']
            }
            rows.append(row)

        return pd.DataFrame(rows)

    @register_plot("Data", "Annotations", FilterParamsDTO)
    def get_annotation_df(self, task_dto: TaskDTO, filter_params: FilterParamsDTO):
        raw = self.get_raw(task_dto, filter_params)
        annots = raw.annotations
        df = pd.DataFrame({
            "onset": annots.onset,
            "duration": annots.duration,
            "description": annots.description
        })
        return df

    @register_plot("Data", "Metadata", None)
    def show_annotations(self, task_dto: TaskDTO, params: FilterParamsDTO):
        task_model = self.get_task(task_dto)
        return task_model.metadata
