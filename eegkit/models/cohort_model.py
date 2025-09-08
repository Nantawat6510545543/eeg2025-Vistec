from typing import List, Optional
import pandas as pd
from mne import concatenate_raws, concatenate_epochs

from .dtos import TaskDTO, SubjectFilterDTO, FilterParamsDTO, EpochParamsDTO
from .task_model import EEGTaskModel
from tqdm.auto import tqdm 


class EEGCohortModel:
    def __init__(self, task_dto: SubjectFilterDTO, subject_list: List[str], data_dir: str):
        self.task_dto = task_dto
        self.data_dir = data_dir
        self.task_model_list: List[EEGTaskModel] = []
        self.filtered_raw = None
        self.epochs = None
        self.labels: Optional[List[str]] = None

        events_list = []
        first_ok_model: Optional[EEGTaskModel] = None
        self.channels = None
        self.electrodes = None
        self.metadata = None

        for subject in tqdm(subject_list,
                        total=len(subject_list),
                        desc="Loading task models",
                        leave=False):
            for run in (None, 1, 2, 3):
                per_subj_dto = TaskDTO(subject=subject, task=task_dto.task, run=run)
                try:
                    task_model = EEGTaskModel(per_subj_dto, data_dir)
                    self.task_model_list.append(task_model)
                    if task_model.events is not None:
                        events_list.append(task_model.events)
                    if first_ok_model is None:
                        first_ok_model = task_model
                except Exception:
                    pass

        self.events = pd.concat(events_list, ignore_index=True) if events_list else None

        if first_ok_model is not None:
            self.channels = first_ok_model.channels
            self.electrodes = getattr(first_ok_model, "electrodes", None)
            self.metadata = getattr(first_ok_model, "metadata", None)
            
    def get_filtered_raw(self, filter_params: FilterParamsDTO):
        if self.filtered_raw is not None:
            return self.filtered_raw

        filtered_list = []
        for task_model in tqdm(self.task_model_list,
                    total=len(self.task_model_list),
                    desc="Filtering raws",
                    leave=False):
            try:
                raw = task_model.get_filtered_raw(filter_params)
            except:
                continue
            if raw is not None:
                filtered_list.append(raw)

        if not filtered_list:
            return None

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
            
            try:
                epochs, labels = task_model.get_epochs(epoch_params)
            except:
                continue
            epochs_list.append(epochs)
            if isinstance(labels, str) and labels == "unavailable":
                return None
            labels_union.update(labels)

        if not epochs_list:
            return None

        print(f"concatrnating {len(epochs_list)} epochs")
        self.epochs = concatenate_epochs(epochs_list)
        self.labels = sorted(labels_union)
        return self.epochs, self.labels

