from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class TaskDTO():
    subject: str
    task: str
    run: Optional[str] = None

@dataclass
class FilterParamsDTO():
    l_freq: float = 3.0
    h_freq: float = 35.0


@dataclass
class EpochParamsDTO(FilterParamsDTO):
    tmin: float = 0.0
    tmax: float = 2.4
    stimulus: List[str] = field(default_factory=lambda: [None])


@dataclass
class EpochFullParamsDTO(EpochParamsDTO):
    n_channels: int = 10

@dataclass
class PSDParamsDTO(FilterParamsDTO):
    fmin: float = 3.0
    fmax: float = 35.0
    average: bool = True
    dB: bool = True
    spatial_colors: bool = True

@dataclass
class EpochPSDParamsDTO(PSDParamsDTO, EpochParamsDTO):
    pass

@dataclass
class TimeDomainParamsDTO(FilterParamsDTO):
    duration: float = 10.0
    start: float = 0.0
    n_channels: int = 10


@dataclass
class TableInfoDTO(FilterParamsDTO):
    table_type: List[str] = field(default_factory=lambda: ["events", "channels", "electrodes"])
    rows: int = 10