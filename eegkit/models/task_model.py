from .task_loader import EEGTaskLoader
from .task_processor import EEGTaskProcessor
from .dtos import TaskDTO, FilterParamsDTO, EpochParamsDTO, TableInfoDTO
import pandas as pd

class EEGTaskModel:
    def __init__(self, task_dto: TaskDTO, data_dir):
        self.task_dto = task_dto
        self.loader = EEGTaskLoader(task_dto, data_dir)

        self.raw = self.loader.load_raw()
        self.events = self.loader.load_events()
        self.channels = self.loader.load_channels()
        self.electrodes = self.loader.load_electrodes()
        self.metadata = self.loader.load_metadata()

        self.processor = EEGTaskProcessor(self.raw, self.events, task_dto.task)

    def get_filtered_raw(self, filter_params: FilterParamsDTO):
        return self.processor.get_filtered(filter_params)

    def get_epochs(self, epoch_params: EpochParamsDTO):
        return self.processor.get_epochs(epoch_params)

