"""Lightweight interfaces and protocols used across EEG models."""

from __future__ import annotations

from typing import Protocol

from .dtos import FilterParamsDTO, EpochParamsDTO


class TaskLike(Protocol):
    """Protocol for task-like models exposing processing methods."""

    def get_filtered_raw(self, filter_params: FilterParamsDTO):
        """Return MNE Raw after applying filters and notch."""
        ...

    def get_epochs(self, epoch_params: EpochParamsDTO):
        """Return (Epochs, labels) for the given parameters."""
        ...

    def get_evoked(self, epoch_params: EpochParamsDTO):
        """Return Evoked (or None) derived from epochs."""
        ...
