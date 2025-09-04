from dataclasses import dataclass, field, fields
from typing import Optional, List, Tuple, ClassVar

NumberRange = Tuple[float, float]

def make_hashable(value):
    if isinstance(value, list):
        return tuple(make_hashable(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in value.items()))
    if isinstance(value, set):
        return frozenset(make_hashable(v) for v in value)
    return value

@dataclass
class BaseTaskDTO:
    task: str
    ui_name: ClassVar[str] = "Base"
    ui_value: ClassVar[object] = None

    def __hash__(self):
        return hash(tuple(make_hashable(getattr(self, f.name)) for f in fields(self)))


@dataclass
class TaskDTO(BaseTaskDTO):
    subject: str
    run: Optional[str] = None

    ui_name: ClassVar[str] = "Single subject"
    ui_value: ClassVar[object] = None

    __hash__ = BaseTaskDTO.__hash__

    def __repr__(self) -> str:
        return f"subject = {self.subject}, task = {self.task}, run = {self.run}"


@dataclass
class SubjectFilterDTO(BaseTaskDTO):
    age_range: NumberRange = (5.0, 21.0)
    sex: List[Optional[str]] = field(default_factory=lambda: [None, "M", "F"])
    ehq_total_range: NumberRange = (-100.0, 100.0)
    p_factor_range: NumberRange = (-10, 10)
    attention_range: NumberRange = (-10, 10)
    internalizing_range: NumberRange = (-10, 10)
    externalizing_range: NumberRange = (-10, 10)

    ui_name: ClassVar[str] = "Meta filter (group)"
    ui_value: ClassVar[object] = None

    __hash__ = BaseTaskDTO.__hash__

    def __repr__(self) -> str:
        return (
            f"task = {self.task}, "
            f"age_range = {self.age_range}, sex = {self.sex}, "
            f"ehq_total_range = {self.ehq_total_range}, p_factor_range = {self.p_factor_range}, "
            f"attention_range = {self.attention_range}, "
            f"internalizing_range = {self.internalizing_range}, "
            f"externalizing_range = {self.externalizing_range}"
        )


@dataclass
class FilterParamsDTO:
    l_freq: float = 3.0
    h_freq: float = 35.0
    ch_min: int = 1
    ch_max: int = 128


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
