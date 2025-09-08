from .task_loader import EEGTaskLoader
from .task_processor import EEGTaskProcessor
from .dtos import TaskDTO, FilterParamsDTO, EpochParamsDTO
from ..cache import LocalCache


class EEGTaskModel:
    def __init__(self, task_dto: TaskDTO, data_dir):
        self.task_dto = task_dto
        self._electrodes = None
        self._metadata = None
        self._channels = None
        self.loader = None
        self._saw = None

        self.loader = EEGTaskLoader(task_dto, data_dir)
        self.events = self.loader.load_events()
        self.cache = LocalCache(data_files=[self.loader.get_file("eeg.set")], pipeline_ver="v1")
        self.processor = EEGTaskProcessor(self.get_raw, self.events, task_dto, self.cache)

    def get_raw(self):
        if not self._saw:
            self._saw = self.loader.load_raw()
        return self._saw

    @property
    def electrodes(self):
        if not self._electrodes:
            self._electrodes = self.loader.load_electrodes()
        return self._electrodes

    @property
    def metadata(self):
        if not self._metadata:
            self._metadata = self.loader.load_metadata()
        return self._metadata

    @property
    def channels(self):
        if not self._channels:
            self._channels = self.loader.load_channels()
        return self._channels

    def get_filtered_raw(self, filter_params: FilterParamsDTO):
        return self.processor.get_filtered(filter_params)

    def get_epochs(self, epoch_params: EpochParamsDTO):
        return self.processor.get_epochs(epoch_params)
