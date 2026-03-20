"""High-level orchestration layer bridging models and services for UI actions."""

from pathlib import Path

from ..models.dtos import (
    FilterParamsDTO, EpochParamsDTO, BaseTaskDTO, SubjectFilterDTO
)
from ..services.ml_service import EEGMLService
from ..services.dl_service import EEGDLService
from ..services.data_service import EEGDataService
from ..services.grid_visualization import EEGGridVisualization
from ..services.visualization import EEGVisualization


class EEGController:
    """Facade providing filtered data, epochs, evoked responses and service specs."""

    def __init__(self, subject_model, jobs_root: Path = None):
        """Initialize controller with subject model and bind service instances."""
        self.subject_model = subject_model
        self.jobs_root = Path(jobs_root or "jobs")
        self.jobs_root.mkdir(exist_ok=True, parents=True)
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
        self.ml_service = EEGMLService(
            get_raw_func=self.get_filtered_raw,
            get_epochs_func=self.get_epochs,
            get_task_func=self.subject_model.get_task,
            get_subjects_metadata_func=self.subject_model.get_subjects_metadata,
            jobs_root=self.jobs_root,
        )
        self.dl_service = EEGDLService(
            get_raw_func=self.get_filtered_raw,
            get_epochs_func=self.get_epochs,
            get_task_func=self.subject_model.get_task,
            get_subjects_metadata_func=self.subject_model.get_subjects_metadata,
            jobs_root=self.jobs_root,
        )

        self._modes = {
            "Plot": {
                "description": getattr(self.visualizer, 'description', "Single-figure visualizations."),
                "spec": self.visualizer.spec,
            },
            # "Grid Plot": {
            #     "description": getattr(self.grid_visualizer, 'description', "Grid-based visualizations."),
            #     "spec": self.grid_visualizer.spec,
            # },
            # "Data": {
            #     "description": getattr(self.data_service, 'description', "Data tables and exports."),
            #     "spec": self.data_service.spec,
            # },
            "Machine Learning": {
                "description": getattr(self.ml_service, 'description', "Machine learning training and inference."),
                "spec": self.ml_service.spec,
            },
            "Deep Learning": {
                "description": getattr(self.dl_service, 'description', "Deep learning training and inference."),
                "spec": self.dl_service.spec,
            }
        }

    def get_filtered_raw(self, task_dto: BaseTaskDTO, filter_params: FilterParamsDTO):
        """Return filtered Raw for given task using provided filter params (with caching)."""
        task_model = self.subject_model.get_task(task_dto)
        return task_model.get_filtered_raw(filter_params)

    def get_epochs(self, task_dto: BaseTaskDTO, epoch_params: EpochParamsDTO):
        """Return (epochs, labels) tuple built from filtered raw for task."""
        task_model = self.subject_model.get_task(task_dto)
        return task_model.get_epochs(epoch_params)

    def get_evoked(self, task_dto: BaseTaskDTO, epoch_params: EpochParamsDTO):
        """Return evoked response (grand-average if cohort) for task."""
        task_model = self.subject_model.get_task(task_dto)
        return task_model.get_evoked(epoch_params)

    def list_subjects(self):
        """List all available subject IDs."""
        return self.subject_model.list_subjects()

    def list_all_tasks(self):
        """List all (subject, task, run) tuples across dataset."""
        return self.subject_model.list_all_tasks()

    def list_tasks(self, subject):
        """List (task, run) entries for a single subject."""
        return self.subject_model.list_tasks(subject)

    def get_specs(self):
        """Return mode -> spec dict describing available actions for UI."""
        return {k: v["spec"] for k, v in self._modes.items()}

    def get_modes_info(self):
        """Return dict of mode -> description for UI display."""
        return {k: v.get("description", "") for k, v in self._modes.items()}

    def prepare(self, task_dto: BaseTaskDTO, group: str, key: str, params_dto):
        """Return parameter enrichment (dropdown choices etc.) for an action."""
        service_map = {
            "Plot": self.visualizer,
            "Grid Plot": self.grid_visualizer,
            "Data": self.data_service,
            "Machine Learning": self.ml_service,
            "Deep Learning": self.dl_service,
        }
        service = service_map.get(group)
        if service is not None:
            return service.prepare_params(task_dto, params_dto)
        return {}

    def show(self, task_dto: BaseTaskDTO, group: str, key: str, params_dto):
        """Execute a visualization/data/AI action and return its result."""
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
