import pandas as pd
from ..models import (
    BaseTaskDTO, FilterParamsDTO, EpochParamsDTO, TableInfoDTO
)

data_registry = {}


def register_data(name, dto_cls):
    def decorator(func):
        data_registry[name] = {
            "params": dto_cls,
            "function": func
        }
        return func

    return decorator


class EEGDataService:
    def __init__(self, get_raw_func, get_epochs_func, get_task_func):
        self.get_raw = get_raw_func
        self.get_epochs = get_epochs_func
        self.get_task = get_task_func
        self.spec = data_registry
        for key in self.spec:
            func = self.spec[key]["function"]
            self.spec[key]["function"] = func.__get__(self)

    @register_data("EEG Table", TableInfoDTO)
    def show_table(self, task_dto: BaseTaskDTO, table_info: TableInfoDTO):
        task_model = self.get_task(task_dto)
        df_map = {
            'events': task_model.events,
            'channels': task_model.channels,
            'electrodes': task_model.electrodes
        }
        return df_map.get(table_info.table_type, pd.DataFrame()).head(table_info.rows)

    @register_data("Epochs Table", EpochParamsDTO)
    def show_epochs_table(self, task_dto: BaseTaskDTO, params: EpochParamsDTO):
        epochs, labels = self.get_epochs(task_dto, params)
        if epochs is None or labels is None:
            return None
        labels_series = pd.Series(labels)
        unique_labels = sorted(set(labels_series))
        rows = []
        for label in unique_labels:
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

    @register_data("Annotations", FilterParamsDTO)
    def get_annotation_df(self, task_dto: BaseTaskDTO, filter_params: FilterParamsDTO):
        raw = self.get_raw(task_dto, filter_params)
        annots = raw.annotations
        df = pd.DataFrame({
            "onset": annots.onset,
            "duration": annots.duration,
            "description": annots.description
        })
        return df

    @register_data("Metadata", None)
    def show_annotations(self, task_dto: BaseTaskDTO, params: FilterParamsDTO):
        task_model = self.get_task(task_dto)
        return task_model.metadata
