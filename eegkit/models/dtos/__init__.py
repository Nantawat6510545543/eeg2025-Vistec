"""DTO package assembling task, filtering, epoching, evoked, table, and AI params.

This package replaces the former monolithic `dtos.py`. All classes are re-exported
for backward compatibility with existing imports like:
    from eegkit.models.dtos import EpochParamsDTO

Additional internal grouping:
- base: task selectors, cohort filters, repr mixin
- filter: preprocessing, epoching, PSD, evoked, time-domain, table
- ai: training/prediction parameter DTOs
"""

from .base import (
    NumberRange,
    make_hashable,
    BaseTaskDTO,
    TaskDTO,
    SubjectFilterDTO,
    ReprMixin,
)
from .filter import (
    FilterParamsDTO,
    EpochParamsDTO,
    PSDParamsDTO,
    EpochPSDParamsDTO,
    EvokedParamsDTO,
    EvokedTopoParamsDTO,
    EvokedJointParamsDTO,
    TimeDomainParamsDTO,
    TableInfoDTO,
    EpochFeatureDatasetParamsDTO,
)
from .ai import (
    DLTrainParamsDTO,
    MLTrainParamsDTO,
    EEGNetMultiRegTrainParamsDTO,
    MLTrainDatasetParamsDTO,
    MLTestDatasetModelParamsDTO,
)

__all__ = [
    # base
    "NumberRange",
    "make_hashable",
    "BaseTaskDTO",
    "TaskDTO",
    "SubjectFilterDTO",
    "ReprMixin",
    # filter
    "FilterParamsDTO",
    "EpochParamsDTO",
    "PSDParamsDTO",
    "EpochPSDParamsDTO",
    "EvokedParamsDTO",
    "EvokedTopoParamsDTO",
    "EvokedJointParamsDTO",
    "TimeDomainParamsDTO",
    "TableInfoDTO",
    "EpochFeatureDatasetParamsDTO",
    # ai
    "DLTrainParamsDTO",
    "MLTrainParamsDTO",
    "EEGNetMultiRegTrainParamsDTO",
    "MLTrainDatasetParamsDTO",
    "MLTestDatasetModelParamsDTO",
]
