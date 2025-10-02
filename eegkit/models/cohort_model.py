import pandas as pd
from mne import concatenate_raws, concatenate_epochs, grand_average

from .dtos import SubjectFilterDTO, FilterParamsDTO, EpochParamsDTO
from .task_model import EEGTaskModel
from tqdm.auto import tqdm
from .interfaces import TaskLike
import logging
import time


class EEGCohortModel(TaskLike):
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
        self._log = logging.getLogger(__name__)

    @property
    def events(self):
        if self._events_concat is None:
            t0 = time.perf_counter()
            events_list = []
            for tm in tqdm(self.task_model_list, total=len(self.task_model_list), desc="Collect events", leave=False):
                ev = tm.get_event()
                if ev is not None:
                    events_list.append(ev)
            self._events_concat = pd.concat(events_list, ignore_index=True) if events_list else pd.DataFrame()
            self._log.info("Concatenated events from %d tasks in %.2fs", len(events_list), time.perf_counter() - t0)
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
        t0 = time.perf_counter()
        for task_model in tqdm(self.task_model_list,
                               total=len(self.task_model_list),
                               desc="Filtering raws",
                               leave=False):
            raw = task_model.get_filtered_raw(filter_params)
            if raw is not None:
                filtered_list.append(raw)
        t_loop = time.perf_counter() - t0
        if not filtered_list:
            self._log.info("No raws produced (loop took %.2fs)", t_loop)
            return None

        self._log.info("Concatenating %d raws (filter loop %.2fs)", len(filtered_list), t_loop)
        t_cat0 = time.perf_counter()
        self.filtered_raw = concatenate_raws(filtered_list)
        t_cat = time.perf_counter() - t_cat0
        self._log.info("Concatenated raws in %.2fs", t_cat)
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

        t0 = time.perf_counter()
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
        t_loop = time.perf_counter() - t0

        if not epochs_list:
            self._log.info("No epochs produced (loop took %.2fs)", t_loop)
            return None

        self._log.info("Concatenating %d epochs (build loop %.2fs)", len(epochs_list), t_loop)
        t_cat0 = time.perf_counter()
        self.epochs = concatenate_epochs(epochs_list)
        t_cat = time.perf_counter() - t_cat0
        self._log.info("Concatenated epochs in %.2fs", t_cat)
        epochs_list.clear()
        self.labels = sorted(labels_union)
        return self.epochs, self.labels

    def get_evoked(self, epoch_params: EpochParamsDTO):
        if self.evoked is not None:
            return self.evoked

        # 1) Get evoked for each task_model
        evokeds_by_subject: dict[str, list] = {}
        t0 = time.perf_counter()
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
        t_evoked_loop = time.perf_counter() - t0

        if not evokeds_by_subject:
            self._log.info("No evoked responses computed (loop took %.2fs)", t_evoked_loop)
            return None

        # 2) For same subject: average runs first (nave-weighted)
        per_subject_evoked = []
        t_sub0 = time.perf_counter()
        for subj, evk_list in evokeds_by_subject.items():
            if not evk_list:
                continue
            if len(evk_list) == 1:
                per_subject_evoked.append(evk_list[0])
            else:
                per_subject_evoked.append(grand_average(evk_list))
        t_sub = time.perf_counter() - t_sub0
        self._log.info("Per-subject averaging: %d subjects in %.2fs", len(evokeds_by_subject), t_sub)

        if not per_subject_evoked:
            return None

        # 3) Grand-average across subjects (nave-weighted)
        t_ga0 = time.perf_counter()
        if len(per_subject_evoked) == 1:
            self.evoked = per_subject_evoked[0]
        else:
            self.evoked = grand_average(per_subject_evoked)
        t_ga = time.perf_counter() - t_ga0
        self._log.info("Grand average across %d subjects in %.2fs (compute loop %.2fs)",
                       len(per_subject_evoked), t_ga, t_evoked_loop)

        return self.evoked
