"""Composite task model wrapping loader + processor with lazy caching."""

from typing import Optional, Type

from .task_loader import EEGTaskLoader
from .task_processor import EEGTaskProcessor
from ..dtos import TaskDTO, FilterParamsDTO, EpochParamsDTO
from ..interfaces import TaskLike
from ...cache import LocalCache, PIPELINE_VERSION


class EEGTaskModel(TaskLike):
    """Convenience wrapper exposing raw, events and processed derivatives for a task."""

    def __init__(
            self,
            task_dto: TaskDTO,
            data_dir,
            *,
            cache: Optional[LocalCache] = None,
            loader_class: Type[EEGTaskLoader] = EEGTaskLoader,
            processor_class: Type[EEGTaskProcessor] = EEGTaskProcessor,
    ):
        """Bind DTO, data path and lazy loader/processor/cache implementations."""
        self.task_dto = task_dto
        self._data_dir = data_dir
        self._electrodes = None
        self._metadata = None
        self._channels = None
        self._raw = None
        self._events = None
        self.loader = None
        self.cache = cache
        self.processor: Optional[EEGTaskProcessor] = None
        self._loader_class = loader_class
        self._processor_class = processor_class

    def _ensure_loader(self):
        """Ensure loader, cache and processor are initialized."""
        if self.loader is None:
            self.loader = self._loader_class(self.task_dto, self._data_dir)
        if self.cache is None:
            self.cache = LocalCache(pipeline_ver=PIPELINE_VERSION)
        if self.processor is None:
            self.processor = self._processor_class(self.get_raw, self.get_event, self.task_dto, self.cache)

    def get_raw(self):
        """Return (and cache) raw MNE object for this task."""
        self._ensure_loader()
        if not self._raw:
            self._raw = self.loader.load_raw()
        return self._raw

    def get_event(self):
        """Return (and cache) events DataFrame for this task."""
        self._ensure_loader()
        if not self._events:
            self._events = self.loader.load_events()
        return self._events

    @property
    def electrodes(self):
        """Return electrodes table for this task (lazy)."""
        self._ensure_loader()
        if not self._electrodes:
            self._electrodes = self.loader.load_electrodes()
        return self._electrodes

    @property
    def metadata(self):
        """Return metadata dict for this task (lazy)."""
        self._ensure_loader()
        if not self._metadata:
            self._metadata = self.loader.load_metadata()
        return self._metadata

    @property
    def channels(self):
        """Return channels table for this task (lazy)."""
        self._ensure_loader()
        if not self._channels:
            self._channels = self.loader.load_channels()
        return self._channels

    def get_filtered_raw(self, filter_params: FilterParamsDTO):
        """Return filtered Raw from processor and clear base raw cache."""
        self._ensure_loader()
        assert self.processor is not None
        raw = self.processor.get_filtered(filter_params)
        self._raw = None
        return raw

    def get_epochs(self, epoch_params: EpochParamsDTO):
        """Return (epochs, labels) from processor for given params."""
        self._ensure_loader()
        assert self.processor is not None
        return self.processor.get_epochs(epoch_params)

    def get_evoked(self, epoch_params: EpochParamsDTO):
        """Return evoked response from processor for given params."""
        self._ensure_loader()
        assert self.processor is not None
        return self.processor.get_evoked(epoch_params)
