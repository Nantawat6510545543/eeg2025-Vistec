"""Subject-level access layer resolving single tasks or building cohorts."""

from __future__ import annotations

import logging
import time

from .cohort_model import EEGCohortModel
from .dtos import BaseTaskDTO, TaskDTO, SubjectFilterDTO
from .interfaces import TaskLike
from .participant_manager import ParticipantManager
from .pipeline.task_model import EEGTaskModel


class EEGSubjectModel:
    """Lookup and construct task models for single subjects or filtered cohorts."""

    def __init__(self, data_dir):
        """Initialize participant index from the given data directory."""
        self._log = logging.getLogger(__name__)
        self.data_dir = data_dir
        t0 = time.perf_counter()
        self._participants = ParticipantManager(data_dir)
        t1 = time.perf_counter()
        self._cache = {}
        self._log.info("EEGSubjectModel initialized (ParticipantManager: %.2fs)", (t1 - t0))

    def list_subjects(self):
        """List all subject IDs discovered in participants table."""
        return self._participants.list_subjects()

    def list_all_tasks(self):
        """List unique task names across the dataset."""
        return self._participants.list_all_tasks()

    def list_tasks(self, subject):
        """List (task, run) entries available for a specific subject."""
        return self._participants.list_tasks(subject)

    def get_task(self, task_dto: BaseTaskDTO) -> TaskLike:
        """Return a task-like model (single subject or cohort) for the DTO."""
        # single
        if getattr(task_dto, "subject", None):
            subj_dir = self._participants.subject_data_dir(task_dto.subject)
            return EEGTaskModel(task_dto, subj_dir)

        # cohort
        subjects = self._participants.filter_subjects_by_dto(task_dto)
        task = task_dto.task
        models = []
        for subj in subjects:
            subj_tasks = self._participants.list_tasks(subj)
            runs = [r for (t, r) in subj_tasks if t == task] or [None]
            subj_dir = self._participants.subject_data_dir(subj)
            for run in runs:
                dto = TaskDTO(subject=subj, task=task, run=run)
                models.append(EEGTaskModel(dto, subj_dir))

        cohort = EEGCohortModel(task_dto, models, len(subjects))
        return cohort

    def get_filter_subjects_dto(self, task_dto: SubjectFilterDTO):
        """Expand a cohort DTO into a list of single-subject TaskDTOs."""
        subjects = self._participants.filter_subjects_by_dto(task_dto)
        task = task_dto.task
        dtos = []
        for subj in subjects:
            subj_tasks = self._participants.list_tasks(subj)
            runs = [r for (t, r) in subj_tasks if t == task] or [None]
            for run in runs:
                dtos.append(TaskDTO(subject=subj, task=task, run=run))

        return dtos

    def get_subjects_metadata(self, subject_ids, columns=None):
        """Return metadata rows for provided subject IDs (optionally subset columns)."""
        return self._participants.get_subjects_metadata(subject_ids, columns)
