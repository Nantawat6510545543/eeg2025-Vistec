from __future__ import annotations

from typing import Protocol

from .dtos import FilterParamsDTO, EpochParamsDTO


class TaskLike(Protocol):
    def get_filtered_raw(self, filter_params: FilterParamsDTO):
        """
        Return an MNE Raw after applying filters and notch.
        """
        ...

    def get_epochs(self, epoch_params: EpochParamsDTO):
        """
        Return (Epochs, labels) for the given parameters.
        """
        ...

    def get_evoked(self, epoch_params: EpochParamsDTO):
        """
        Return an Evoked object (or None) derived from epochs.
        """
        ...
