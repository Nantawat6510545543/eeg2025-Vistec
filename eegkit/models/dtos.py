from dataclasses import dataclass, field
from typing import Optional, List, Union, Dict, Any


@dataclass
class SubjectDTO:
    subject_id: str
    tasks: List[Dict[str, Optional[str]]] = field(default_factory=list)


@dataclass
class TaskDTO:
    subject: str
    task: str
    run: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FilterParamsDTO:
    l_freq: float = 3.0
    h_freq: float = 35.0


@dataclass
class EpochParamsDTO(FilterParamsDTO):
    tmin: float = 0.0
    tmax: float = 2.4
    stimulus: Optional[str] = None


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
class TimeDomainParamsDTO(FilterParamsDTO):
    duration: float = 10.0
    start: float = 0.0
    n_channels: int = 10


@dataclass
class TableInfoDTO(FilterParamsDTO):
    table_type: str = "events"
    rows: int = 10
    data: Union[List[Dict[str, Any]], str] = field(default_factory=list)