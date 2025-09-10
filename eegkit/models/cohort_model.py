import pandas as pd
from mne import concatenate_raws, concatenate_epochs

from .dtos import SubjectFilterDTO, FilterParamsDTO, EpochParamsDTO
from .task_model import EEGTaskModel
from tqdm.auto import tqdm


class EEGCohortModel:
    def __init__(self, task_dto: SubjectFilterDTO, task_model_list: list[EEGTaskModel], subject_length: int):
        self.task_dto = task_dto
        self.task_model_list = task_model_list
        self.subject_length = subject_length
        self.filtered_raw = None
        self.epochs = None
        self.labels = None
        self._electrodes = None
        self._metadata = None
        self._channels = None
        self._events_concat = None 

    @property
    def events(self):
        if self._events_concat is None:
            events_list = [tm.get_event() for tm in self.task_model_list]
            self._events_concat = pd.concat(events_list, ignore_index=True)
        return self._events_concat

    @property
    def electrodes(self):
        if not self._electrodes and self.task_model_list:
            self._electrodes = getattr(self.task_model_list[0], "electrodes", None)
        return self._electrodes

    @property
    def metadata(self):
        if not self._metadata and self.task_model_list:
            self._metadata = getattr(self.task_model_list[0], "metadata", None)
        return self._metadata

    @property
    def channels(self):
        if not self._channels and self.task_model_list:
            self._channels = self.task_model_list[0].channels
        return self._channels

    def get_filtered_raw(self, filter_params: FilterParamsDTO):
        if self.filtered_raw is not None:
            return self.filtered_raw

        filtered_list = []
        for task_model in tqdm(self.task_model_list,
                               total=len(self.task_model_list),
                               desc="Filtering raws",
                               leave=False):
            raw = task_model.get_filtered_raw(filter_params)
            if raw is not None:
                filtered_list.append(raw)

        if not filtered_list:
            return None

        print(f"concatenating {len(filtered_list)} raws")
        self.filtered_raw = concatenate_raws(filtered_list)
        return self.filtered_raw

    def get_epochs(self, epoch_params: EpochParamsDTO):
        if self.epochs is not None:
            return self.epochs, self.labels

        epochs_list = []
        labels_union = set()

        for task_model in tqdm(self.task_model_list,
                               total=len(self.task_model_list),
                               desc="Building epochs",
                               leave=False):

            epochs, labels = task_model.get_epochs(epoch_params)
            if epochs is None:
                continue
            epochs_list.append(epochs)
            if isinstance(labels, str) and labels == "unavailable":
                return None
            if labels is not None:
                labels_union.update(labels)

        if not epochs_list:
            return None

        print(f"concatenating {len(epochs_list)} epochs")
        self.epochs = concatenate_epochs(epochs_list)
        self.labels = sorted(labels_union)
        return self.epochs, self.labels
