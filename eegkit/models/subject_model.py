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
        subjects = self._participants.filter_subjects_by_dto(task_dto)
        task = task_dto.task
        dtos = []
        for subj in subjects:
            subj_tasks = self._participants.list_tasks(subj)
            runs = [r for (t, r) in subj_tasks if t == task] or [None]
            for run in runs:
                dtos.append(TaskDTO(subject=subj, task=task, run=run))
                
        return dtos
