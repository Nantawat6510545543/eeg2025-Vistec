import pandas as pd
from mne import concatenate_raws, concatenate_epochs, grand_average

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
        self.evoked = None
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
        try:
            for r in filtered_list:
                if hasattr(r, 'close') and r is not self.filtered_raw:
                    try:
                        r.close()
                    except Exception:
                        pass
        finally:
            filtered_list.clear()
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
        epochs_list.clear()
        self.labels = sorted(labels_union)
        return self.epochs, self.labels

    def get_evoked(self, epoch_params: EpochParamsDTO):
        if self.evoked is not None:
            return self.evoked

        # 1) Get evoked for each task_model
        evokeds_by_subject: dict[str, list] = {}
        for task_model in tqdm(self.task_model_list,
                               total=len(self.task_model_list),
                               desc="Computing evoked",
                               leave=False):
            evk = task_model.get_evoked(epoch_params)
            if evk is None:
                continue
            subj = getattr(task_model.task_dto, 'subject', None)
            if subj is None:
                continue
            evokeds_by_subject.setdefault(subj, []).append(evk)

        if not evokeds_by_subject:
            return None

        # 2) For same subject: average runs first (nave-weighted)
        per_subject_evoked = []
        for subj, evk_list in evokeds_by_subject.items():
            if not evk_list:
                continue
            if len(evk_list) == 1:
                per_subject_evoked.append(evk_list[0])
            else:
                try:
                    per_subject_evoked.append(grand_average(evk_list, weights='nave'))
                except Exception:
                    per_subject_evoked.append(grand_average(evk_list))

        if not per_subject_evoked:
            return None

        # 3) Grand-average across subjects (nave-weighted)
        if len(per_subject_evoked) == 1:
            self.evoked = per_subject_evoked[0]
        else:
            try:
                self.evoked = grand_average(per_subject_evoked, weights='nave')
            except Exception:
                self.evoked = grand_average(per_subject_evoked)

        return self.evoked
