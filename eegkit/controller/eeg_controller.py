from ..models import (
    FilterParamsDTO, EpochParamsDTO, BaseTaskDTO, SubjectFilterDTO
)
from ..services import EEGVisualization, EEGDataService
from ..services.ai_service import EEGAIService
from ..services.grid_visualization import EEGGridVisualization


class EEGController:
    def __init__(self, subject_model):
        self.subject_model = subject_model
        self.visualizer = EEGVisualization(
            get_raw_func=self.get_filtered_raw,
            get_epochs_func=self.get_epochs,
            get_evoked_func=self.get_evoked,
            get_task_func=self.subject_model.get_task
        )
        self.grid_visualizer = EEGGridVisualization(
            get_raw_func=self.get_filtered_raw,
            get_epochs_func=self.get_epochs,
            get_evoked_func=self.get_evoked,
            get_task_func=self.subject_model.get_task
        )
        self.data_service = EEGDataService(
            get_raw_func=self.get_filtered_raw,
            get_epochs_func=self.get_epochs,
            get_task_func=self.subject_model.get_task
        )
        self.ai_service = EEGAIService(
            get_raw_func=self.get_filtered_raw,
            get_epochs_func=self.get_epochs,
            get_task_func=self.subject_model.get_task,
        )

        self._modes = {
            "Plot": {
                "description": getattr(self.visualizer, 'description', "Single-figure visualizations."),
                "spec": self.visualizer.spec,
            },
            "Grid Plot": {
                "description": getattr(self.grid_visualizer, 'description', "Grid-based visualizations."),
                "spec": self.grid_visualizer.spec,
            },
            "Data": {
                "description": getattr(self.data_service, 'description', "Data tables and exports."),
                "spec": self.data_service.spec,
            },
            "AI": {
                "description": getattr(self.ai_service, 'description', "AI training and inference."),
                "spec": self.ai_service.spec,
            },
        }

    def get_filtered_raw(self, task_dto: BaseTaskDTO, filter_params: FilterParamsDTO):
        task_model = self.subject_model.get_task(task_dto)
        return task_model.get_filtered_raw(filter_params)

    def get_epochs(self, task_dto: BaseTaskDTO, epoch_params: EpochParamsDTO):
        task_model = self.subject_model.get_task(task_dto)
        return task_model.get_epochs(epoch_params)

    def get_evoked(self, task_dto: BaseTaskDTO, epoch_params: EpochParamsDTO):
        task_model = self.subject_model.get_task(task_dto)
        return task_model.get_evoked(epoch_params)

    def list_subjects(self):
        return self.subject_model.list_subjects()

    def list_all_tasks(self):
        return self.subject_model.list_all_tasks()

    def list_tasks(self, subject):
        return self.subject_model.list_tasks(subject)

    def get_specs(self):
        return {k: v["spec"] for k, v in self._modes.items()}

    def get_modes_info(self):
        """Return dict of mode -> description for UI display."""
        return {k: v.get("description", "") for k, v in self._modes.items()}

    def prepare(self, task_dto: BaseTaskDTO, group: str, key: str, params_dto):
        if group == "Plot":
            return self.visualizer.prepare_params(task_dto, params_dto)
        if group == "Grid Plot":
            return self.grid_visualizer.prepare_params(task_dto, params_dto)
        if group == "AI":
            return self.ai_service.prepare_params(task_dto, params_dto)
        return {}

    def show(self, task_dto: BaseTaskDTO, group: str, key: str, params_dto):
        # Per-subject expansion mode
        if isinstance(task_dto, SubjectFilterDTO) and getattr(task_dto, 'per_subject', False):
            dtos = self.subject_model.get_filter_subjects_dto(task_dto)
            out = {}
            for single_dto in dtos:
                res = self._modes[group]["spec"][key]["function"](single_dto, params_dto)
                out[single_dto.subject] = res
            return out

        # Default / existing behavior
        result = self._modes[group]["spec"][key]["function"](task_dto, params_dto)
        return result
