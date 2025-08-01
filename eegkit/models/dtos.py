from dataclasses import dataclass
from typing import Optional

def make_hashable(obj):
    if isinstance(obj, (tuple, list)):
        return tuple(make_hashable(e) for e in obj)
    elif isinstance(obj, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, set):
        return frozenset(make_hashable(e) for e in obj)
    return obj

@dataclass
class TaskDTO():
    subject: str
    task: str
    run: Optional[str] = None
    
    def __eq__(self, other):
        return type(self) is type(other) and self.__dict__ == other.__dict__
    def __hash__(self):
        return hash(hash(make_hashable(self.__dict__)))


@dataclass
class FilterParamsDTO():
    l_freq: float = 3.0
    h_freq: float = 35.0
    
    def __eq__(self, other):
        return type(self) is type(other) and self.__dict__ == other.__dict__
    def __hash__(self):
        return hash(hash(make_hashable(self.__dict__)))


@dataclass
class EpochParamsDTO(FilterParamsDTO):
    tmin: float = 0.0
    tmax: float = 2.4
    stimulus: Optional[str] = None

    def __eq__(self, other):
        return type(self) is type(other) and self.__dict__ == other.__dict__
    def __hash__(self):
        return hash(hash(make_hashable(self.__dict__)))


@dataclass
class EpochFullParamsDTO(EpochParamsDTO):
    n_channels: int = 10

    def __eq__(self, other):
        return type(self) is type(other) and self.__dict__ == other.__dict__
    def __hash__(self):
        return hash(hash(make_hashable(self.__dict__)))

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