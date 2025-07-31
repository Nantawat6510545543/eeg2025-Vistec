import matplotlib.pyplot as plt
import pandas as pd
from ..models import TaskDTO, FilterParamsDTO, TimeDomainParamsDTO, PSDParamsDTO, EpochParamsDTO, EpochFullParamsDTO, TableInfoDTO

class EEGVisualization:
    def __init__(self, get_raw_func, get_epochs_func, get_task_func):
        self.get_raw = get_raw_func
        self.get_epochs = get_epochs_func
        self.get_task = get_task_func
        self.specs = {
            "data_specs": {
                "meta": {
                    "label": "Metadata",
                    "params": None,
                    "function": self.show_annotations
                },
                "table": {
                    "label": "EEG Table",
                    "params": TableInfoDTO,
                    "function": self.show_table
                },
                "annotations": {
                    "label": "get_annotation_df",
                    "params": FilterParamsDTO,
                    "function": self.get_annotation_df
                }
            },
            "plot_specs": {
                "sensors": {
                    "label": "Sensor Layout",
                    "params": FilterParamsDTO,
                    "function": self.plot_sensors
                },
                "time": {
                    "label": "Time Domain Plot",
                    "params": TimeDomainParamsDTO,
                    "function": self.plot_time
                },
                "frequency": {
                    "label": "Frequency Domain",
                    "params": PSDParamsDTO,
                    "function": self.plot_frequency
                },
                "conditionwise_psd": {
                    "label": "Condition-wise PSD",
                    "params": EpochParamsDTO,
                    "function": self.plot_conditionwise_psd
                },
                "epochs": {
                    "label": "Epoch Plot",
                    "params": EpochFullParamsDTO,
                    "function": self.plot_epochs
                },
                "evoked": {
                    "label": "Evoked Response",
                    "params": EpochParamsDTO,
                    "function": self.plot_evoked
                }
            }
        }

    def _finalize_figure(self, fig, task_dto: TaskDTO, stimulus=None, caption: dict = None, plot_name="EEG Plot", x=15, y=10):
        if not isinstance(fig, plt.Figure):
            return
        fig.set_size_inches(x, y)
        subject_line = f"{task_dto.subject} - {task_dto.task}" + (f" - {stimulus}" if stimulus else "") + (f" (Run {task_dto.run})" if task_dto.run else "")
        caption_line = ", ".join(f"{k} = {v:.1f}" if isinstance(v, (float, int)) else f"{k} = {v}" for k, v in caption.items()) if caption else ""
        fig.text(0.5, 0.96, plot_name.title(), ha='center', fontsize=18, weight='bold')
        fig.text(0.5, 0.94, subject_line, ha='center', fontsize=14)
        if caption_line:
            fig.text(0.5, 0.92, caption_line, ha='center', fontsize=11)
        fig.subplots_adjust(top=0.90)
        return fig

    def plot_sensors(self, task_dto: TaskDTO, params: FilterParamsDTO):
        raw = self.get_raw(task_dto, params)
        raw.plot_sensors(show_names=True)

    def plot_time(self, task_dto: TaskDTO, params: TimeDomainParamsDTO):
        raw = self.get_raw(task_dto, params)
        fig = raw.plot(
            duration=params.duration,
            start=params.start,
            n_channels=params.n_channels,
            scalings='auto',
            show=False,
            block=True
        )
        self._finalize_figure(fig, task_dto, caption=vars(params), plot_name="Time Domain")

    def plot_frequency(self, task_dto: TaskDTO, params: PSDParamsDTO):
        raw = self.get_raw(task_dto, params)
        psd = raw.compute_psd(fmin=params.fmin, fmax=params.fmax)
        fig = psd.plot(
            average=params.average,
            spatial_colors=params.spatial_colors,
            dB=params.dB,
            show=False
        )
        self._finalize_figure(fig, task_dto, caption=vars(params), plot_name="Frequency Domain")

    def plot_conditionwise_psd(self, task_dto: TaskDTO, params: EpochParamsDTO):
        epochs, labels = self.get_epochs(task_dto, params)
        if epochs is None:
            print(f"No epochs available for {task_dto.subject} - {task_dto.task}")
            return
        for condition in epochs.event_id:
            condition_epochs = epochs[condition]
            if len(condition_epochs) == 0:
                continue
            cropped = condition_epochs.copy().crop(tmin=params.tmin, tmax=params.tmax)
            psd = cropped.compute_psd(fmin=params.fmin, fmax=params.fmax)
            fig = psd.plot(average=params.average, spatial_colors=True, dB=params.dB, show=False)
            self._finalize_figure(fig, task_dto, condition, caption=vars(params), plot_name="Condition-wise PSD")

    def plot_epochs(self, task_dto: TaskDTO, params: EpochParamsDTO):
        epochs, labels = self.get_epochs(task_dto, params)
        if labels is not None:
            pass
            # self.plot_specs["epochs"]["params"]["stimulus"]["default"] = [None] + sorted(labels)
        if epochs is None:
            return
        if params.stimulus and params.stimulus in epochs.event_id:
            epochs = epochs[params.stimulus]
        cropped = epochs.copy().crop(tmin=params.tmin, tmax=params.tmax)
        fig = cropped.plot(events=False, n_channels=params.n_channels, show=False)
        return self._finalize_figure(fig, task_dto, params.stimulus, caption=vars(params), plot_name="Epochs")

    def plot_evoked(self, task_dto: TaskDTO, params: EpochParamsDTO):
        epochs, labels = self.get_epochs(task_dto, params)
        if labels is not None:
            pass
            # self.plot_specs["evoked"]["params"]["stimulus"]["default"] = [None] + sorted(labels)
        if epochs is None:
            return
        if params.stimulus and params.stimulus in epochs.event_id:
            epochs = epochs[params.stimulus]
        cropped = epochs.copy().crop(tmin=params.tmin, tmax=params.tmax)
        fig = cropped.average().plot_joint(show=False)
        return self._finalize_figure(fig, task_dto, params.stimulus, caption=vars(params), plot_name="Evoked")

    def show_table(self, task_dto: TaskDTO, table_info: TableInfoDTO):
        task_model = self.get_task(task_dto)

        if table_info.table_type == 'epochs':
            epochs, labels = self.get_epochs(task_dto, table_info)
            if epochs is None:
                return None
            info = {
                'n_epochs': len(epochs),
                'n_channels': len(epochs.ch_names),
                'timespan_sec': epochs.times[-1] - epochs.times[0],
                'labels': sorted(set(labels)) if labels is not None else 'N/A',
                'sampling_rate': epochs.info['sfreq'],
                'duration_per_epoch_sec': epochs.get_data().shape[-1] / epochs.info['sfreq']
            }
            return pd.DataFrame([info])

        df_map = {
            'events': task_model.events,
            'channels': task_model.channels,
            'electrodes': task_model.electrodes
        }
        return df_map.get(table_info.table_type, pd.DataFrame()).head(table_info.rows)
    
    def get_annotation_df(self, task_dto: TaskDTO, filter_params: FilterParamsDTO):
        raw = self.get_raw_func(filter_params)
        annots = raw.annotations
        df = pd.DataFrame({
            "onset": annots.onset,
            "duration": annots.duration,
            "description": annots.description
        })
        return df

    def show_annotations(self, task_dto: TaskDTO, params: FilterParamsDTO):
        task_model = self.get_task(task_dto)
        return task_model.metadata

