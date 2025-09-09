from dataclasses import dataclass, field, fields
from typing import Optional, List, Tuple, ClassVar
import re

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
    task: str
    subject: str
    run: Optional[str] = None

    ui_name: ClassVar[str] = "Single subject"
    ui_value: ClassVar[object] = None

    __hash__ = BaseTaskDTO.__hash__

    def __repr__(self) -> str:
        return f"subject = {self.subject}, task = {self.task}, run = {self.run}"


@dataclass
class SubjectFilterDTO(BaseTaskDTO):
    task: str
    age_range: NumberRange = (5.0, 6.0)
    sex: List[Optional[str]] = field(default_factory=lambda: [None, "M", "F"])
    ehq_total_range: NumberRange = (-1000.0, 1000.0)
    p_factor_range: NumberRange = (-100, 100)
    attention_range: NumberRange = (-100, 100)
    internalizing_range: NumberRange = (-100, 100)
    externalizing_range: NumberRange = (-100, 100)

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
    channels: str = "70, 71, 74, 75, 76, 81, 82, 83"

    @property
    def channels_list(self):
        if not self.channels or not self.channels.strip():
            return [i for i in range(1, 129)]

        # Match "70" or "70-90"
        number_or_range = re.compile(r'^(\d+)(?:\s*-\s*(\d+))?$')

        # Split on commas or whitespace
        raw_tokens = re.split(r'[,\s]+', self.channels.strip())

        seen_channel_names = set()
        parsed_channel_names: list[int] = []

        def add_channel_name(channel_name: int) -> None:
            if channel_name not in seen_channel_names:
                seen_channel_names.add(channel_name)
                parsed_channel_names.append(channel_name)

        for token in raw_tokens:
            if not token:
                continue

            match = number_or_range.match(token)
            if not match:
                continue

            start_number_str, end_number_str = match.groups()
            start_number = int(start_number_str) - 1

            if end_number_str is None:
                add_channel_name(start_number)
            else:
                end_number = int(end_number_str) - 1
                lower_bound, upper_bound = sorted((start_number, end_number))
                for number in range(lower_bound, upper_bound + 1):
                    add_channel_name(number)

        return parsed_channel_names or [i for i in range(1, 129)]


@dataclass
class EpochParamsDTO(FilterParamsDTO):
    tmin: float = 0.0
    tmax: float = 2.4
    stimulus: List[str] = field(default_factory=lambda: [None])
    only_labels: ClassVar[bool] = False


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
