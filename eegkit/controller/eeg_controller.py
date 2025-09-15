from ..models import (
    FilterParamsDTO, EpochParamsDTO, BaseTaskDTO
)
from ..services import EEGVisualization, EEGDataService


class EEGController:
    def __init__(self, subject_model):
        self.subject_model = subject_model
        self.visualizer = EEGVisualization(
            get_raw_func=self.get_filtered_raw,
            get_epochs_func=self.get_epochs,
            get_task_func=self.subject_model.get_task
        )
        self.data_service = EEGDataService(
            get_raw_func=self.get_filtered_raw,
            get_epochs_func=self.get_epochs,
            get_task_func=self.subject_model.get_task
        )

        self.specs = {
            "plot": self.visualizer.spec,
            "data": self.data_service.spec,
        }

    def get_filtered_raw(self, task_dto: BaseTaskDTO, filter_params: FilterParamsDTO):
        task_model = self.subject_model.get_task(task_dto)
        return task_model.get_filtered_raw(filter_params)

    def get_epochs(self, task_dto: BaseTaskDTO, epoch_params: EpochParamsDTO):
        task_model = self.subject_model.get_task(task_dto)
        return task_model.get_epochs(epoch_params)

    def list_subjects(self):
        return self.subject_model.list_subjects()

    def list_all_tasks(self):
        return self.subject_model.list_all_tasks()

    def list_tasks(self, subject):
        return self.subject_model.list_tasks(subject)

    def get_specs(self):
        return self.specs

    def prepare(self, task_dto: BaseTaskDTO, group: str, key: str):
        if group == "plot":
            return self.visualizer.prepare_params(task_dto, key)
        else:
            return {}

    def show(self, task_dto: BaseTaskDTO, group: str, key: str, params_dto):
        result = self.specs[group][key]["function"](task_dto, params_dto)
        return result
