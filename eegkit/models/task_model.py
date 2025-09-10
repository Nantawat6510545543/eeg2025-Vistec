from .task_loader import EEGTaskLoader
from .task_processor import EEGTaskProcessor
from .dtos import TaskDTO, FilterParamsDTO, EpochParamsDTO
from ..cache import LocalCache


class EEGTaskModel:
    def __init__(self, task_dto: TaskDTO, data_dir):
        self.task_dto = task_dto
        self._data_dir = data_dir
        self._electrodes = None
        self._metadata = None
        self._channels = None
        self._saw = None
        self._events = None
        self.loader = None
        self.cache = None
        self.processor = None

    def _ensure_loader(self):
        if self.loader is None:
            self.loader = EEGTaskLoader(self.task_dto, self._data_dir)
        if self.cache is None:
            self.cache = LocalCache(pipeline_ver="v1")
        if self.processor is None:
            self.processor = EEGTaskProcessor(self.get_raw, self.get_event, self.task_dto, self.cache)

    def get_raw(self):
        self._ensure_loader()
        if not self._saw:
            self._saw = self.loader.load_raw()
        return self._saw

    def get_event(self):
        self._ensure_loader()
        if not self._events:
            self._events = self.loader.load_events()
        return self._events

    @property
    def electrodes(self):
        self._ensure_loader()
        if not self._electrodes:
            self._electrodes = self.loader.load_electrodes()
        return self._electrodes

    @property
    def metadata(self):
        self._ensure_loader()
        if not self._metadata:
            self._metadata = self.loader.load_metadata()
        return self._metadata

    @property
    def channels(self):
        self._ensure_loader()
        if not self._channels:
            self._channels = self.loader.load_channels()
        return self._channels

    def get_filtered_raw(self, filter_params: FilterParamsDTO):
        self._ensure_loader()
        return self.processor.get_filtered(filter_params)

    def get_epochs(self, epoch_params: EpochParamsDTO):
        self._ensure_loader()
        return self.processor.get_epochs(epoch_params)
