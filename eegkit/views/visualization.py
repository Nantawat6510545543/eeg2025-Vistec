import pandas as pd
from ..models import TaskDTO, FilterParamsDTO, TimeDomainParamsDTO, PSDParamsDTO, EpochParamsDTO, EpochFullParamsDTO, TableInfoDTO, EpochPSDParamsDTO
import matplotlib.pyplot as plt
plt.ioff()

class EEGVisualization:
    def __init__(self, get_raw_func, get_epochs_func, get_task_func):
        self.get_raw = get_raw_func
        self.get_epochs = get_epochs_func
        self.get_task = get_task_func
        self.specs = {
            "Plot": {
                "Sensor Layout": {
                    "params": FilterParamsDTO(),
                    "function": self.plot_sensors
                },
                "Time Domain Plot": {
                    "params": TimeDomainParamsDTO(),
                    "function": self.plot_time
                },
                "Frequency Domain": {
                    "params": PSDParamsDTO(),
                    "function": self.plot_frequency
                },
                "Condition-wise PSD": {
                    "params": EpochPSDParamsDTO(),
                    "function": self.plot_conditionwise_psd
                },
                "Epoch Plot": {
                    "params": EpochFullParamsDTO(),
                    "function": self.plot_epochs
                },
                "Evoked Response": {
                    "params": EpochParamsDTO(),
                    "function": self.plot_evoked
                }
            },
            "Data": {
                "Metadata": {
                    "params": None,
                    "function": self.show_annotations
                },
                "EEG Table": {
                    "params": TableInfoDTO(),
                    "function": self.show_table
                },
                "Annotations": {
                    "params": FilterParamsDTO(),
                    "function": self.get_annotation_df
                }
            },
        }

    def _finalize_figure(self, fig: plt.Figure, task_dto: TaskDTO, stimulus=None, caption: dict = None, plot_name="EEG Plot", x=15, y=10) -> plt.Figure:
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
        )
        fig = self._finalize_figure(fig, task_dto, caption=vars(params), plot_name="Time Domain")
        return [fig]

    def plot_frequency(self, task_dto: TaskDTO, params: PSDParamsDTO):
        raw = self.get_raw(task_dto, params)
        psd = raw.compute_psd(fmin=params.fmin, fmax=params.fmax)
        fig = psd.plot(
            average=params.average,
            spatial_colors=params.spatial_colors,
            dB=params.dB,
            show=False
        )
        fig = self._finalize_figure(fig, task_dto, caption=vars(params), plot_name="Frequency Domain")
        return [fig]

    def plot_conditionwise_psd(self, task_dto: TaskDTO, params: EpochPSDParamsDTO):
        epochs, labels = self.get_epochs(task_dto, params)
        if epochs is None:
            print(f"No epochs available for {task_dto.subject} - {task_dto.task}")
            return
        fig_list =[]
        for condition in epochs.event_id:
            condition_epochs = epochs[condition]
            if len(condition_epochs) == 0:
                continue
            psd = epochs.compute_psd(fmin=params.fmin, fmax=params.fmax)
            fig = psd.plot(average=params.average, spatial_colors=params.spatial_colors, dB=params.dB, show=False)
            fig = self._finalize_figure(fig, task_dto, condition, caption=vars(params), plot_name="Condition-wise PSD")
            fig_list.append(fig)
        return fig_list

    def plot_epochs(self, task_dto: TaskDTO, params: EpochParamsDTO):
        return self._plot_epochs_base(task_dto, params, mode="Epochs")

    def plot_evoked(self, task_dto: TaskDTO, params: EpochParamsDTO):
        return self._plot_epochs_base(task_dto, params, mode="Evoked")

    def _plot_epochs_base(self, task_dto, params, mode):
        epochs, labels = self.get_epochs(task_dto, params)
        if labels is not None:
            self.specs["Plot"]["Epoch Plot"]["params"].stimulus = [None] + sorted(labels)
        if epochs is None:
            print(f"No epochs available for {task_dto.subject} - {task_dto.task}")
            return
        if params.stimulus and params.stimulus in epochs.event_id:
            epochs = epochs[params.stimulus]

        if mode == "Epochs":
            fig = epochs.plot(events=False, n_channels=params.n_channels, show=False)
        elif mode == "Evoked":
            fig = epochs.average().plot_joint(show=False)

        fig = self._finalize_figure(fig, task_dto, params.stimulus, caption=vars(params), plot_name=mode)
        return [fig]

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
        raw = self.get_raw(task_dto, filter_params)
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

