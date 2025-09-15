from .task_model import EEGTaskModel
from .dtos import BaseTaskDTO, TaskDTO, SubjectFilterDTO
from .cohort_model import EEGCohortModel
from .participant_manager import ParticipantManager


class EEGSubjectModel:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self._participants = ParticipantManager(data_dir)
        self._cache = {}

    def list_subjects(self):
        return self._participants.list_subjects()

    def list_all_tasks(self):
        return self._participants.list_all_tasks()

    def list_tasks(self, subject):
        return self._participants.list_tasks(subject)

    def get_task(self, task_dto: BaseTaskDTO):
        # single
        if getattr(task_dto, "subject", None):
            key = ("single", hash(task_dto))
            if key not in self._cache:
                subj_dir = self._participants.subject_data_dir(task_dto.subject)
                self._cache[key] = EEGTaskModel(task_dto, subj_dir)
            return self._cache[key]

        # cohort
        key = ("cohort", hash(task_dto))
        if key in self._cache:
            return self._cache[key]

        subjects = self._participants.filter_subjects_by_dto(task_dto)
        print(f"{len(subjects)} subjects found")
        task = task_dto.task
        models = []
        for subj in subjects:
            subj_tasks = self._participants.list_tasks(subj)
            runs = [r for (t, r) in subj_tasks if t == task] or [None]
            subj_dir = self._participants.subject_data_dir(subj)
            for run in runs:
                dto = TaskDTO(subject=subj, task=task, run=run)
                skey = ("single", hash(dto))
                if skey not in self._cache:
                    self._cache[skey] = EEGTaskModel(dto, subj_dir)
                models.append(self._cache[skey])

        cohort = EEGCohortModel(task_dto, models, len(subjects))
        self._cache[key] = cohort
        return cohort


