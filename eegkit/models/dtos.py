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
    subject_limit: Optional[int] = None 
    age_range: Optional[NumberRange] = (None, None)
    sex: List[Optional[str]] = field(default_factory=lambda: [None, "M", "F"])
    ehq_total_range: Optional[NumberRange] = (None, None)
    p_factor_range: Optional[NumberRange] = (None, None)
    attention_range: Optional[NumberRange] = (None, None)
    internalizing_range: Optional[NumberRange] = (None, None)
    externalizing_range: Optional[NumberRange] = (None, None)
    ccd_accuracy_range: Optional[NumberRange] = (None, None)
    ccd_response_time_range: Optional[NumberRange] = (None, None)

    ui_name: ClassVar[str] = "Meta filter (group)"
    ui_value: ClassVar[object] = None

    __hash__ = BaseTaskDTO.__hash__

    def __repr__(self) -> str:
        def fmt_range(label: str, rng):
            if not (isinstance(rng, (tuple, list)) and len(rng) == 2):
                return None
            lo, hi = rng
            if lo is None and hi is None:
                return None
            def _n(v):
                if v is None:
                    return None
                return int(v) if isinstance(v, (int, float)) and float(v).is_integer() else v
            lo, hi = _n(lo), _n(hi)
            if lo is None:
                return f"{label} <= {hi}"
            if hi is None:
                return f"{label} >= {lo}"
            return f"{label} [{lo}, {hi}]"

        parts = [f"Task={self.task}"]
        if self.subject_limit is not None:
            parts.append(f"Number of subjects={self.subject_limit}")
        if self.sex[0]:
            parts.append(f"sex={self.sex[0]}")

        for label, rng in [
            ("Age", self.age_range),
            ("Ehq total", self.ehq_total_range),
            ("P factor", self.p_factor_range),
            ("Attention", self.attention_range),
            ("Internalizing", self.internalizing_range),
            ("ccd_accuracy", self.ccd_accuracy_range),
            ("ccd_response_time", self.ccd_response_time_range),
        ]:
            fr = fmt_range(label, rng)
            if fr:
                parts.append(fr)

        return ", ".join(parts)


@dataclass
class FilterParamsDTO:
    l_freq: float = 0.5
    h_freq: float = 55.0
    notch: float = 60.0
    channels: str = "69-76,81-83,88,89"
    combine_channels: bool = False

    @property
    def key(self):
        key = {
            "l_freq": self.l_freq,
            "h_freq": self.h_freq,
            "notch": self.notch,
            }
        return key

    @property
    def channels_list(self):
        if not self.channels or not self.channels.strip():
            return [f"E{i}" for i in range(1, 129)]

        number_or_range = re.compile(r'^(\d+)(?:\s*-\s*(\d+))?$')

        raw_tokens = re.split(r'[\s,]+', self.channels.strip())

        seen_channel_names = set()
        parsed_channel_names: list[str] = []

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

            if end_number_str is None:
                add_channel_name(f"E{int(start_number_str)}")
            else:
                start_number = int(start_number_str)
                end_number = int(end_number_str)
                lower_bound, upper_bound = sorted((start_number, end_number))
                for number in range(lower_bound, upper_bound + 1):
                    add_channel_name(f"E{number}")

        return parsed_channel_names or [f"E{i}" for i in range(1, 129)]


@dataclass
class EpochParamsDTO(FilterParamsDTO):
    tmin: float = -0.2
    tmax: float = 2.4
    stimulus: List[str] = field(default_factory=lambda: [None])
    only_labels: ClassVar[bool] = False

    @property
    def key(self):
        key = {
            "l_freq": self.l_freq,
            "h_freq": self.h_freq,
            "notch": self.notch,
            "tmin": self.tmin,
            "tmax": self.tmax,
            }
        return key


@dataclass
class PSDParamsDTO(FilterParamsDTO):
    fmin: float = 3.0
    fmax: float = 55.0
    average: bool = True
    dB: bool = True
    spatial_colors: bool = True


@dataclass
class EpochPSDParamsDTO(PSDParamsDTO, EpochParamsDTO):
    pass

@dataclass
class EvokedParamsDTO(EpochParamsDTO):
    spatial_colors: bool = True
    gfp: List[Optional[str]] = field(default_factory=lambda: [True, False, "only"])

@dataclass
class EvokedTopoParamsDTO(EpochParamsDTO):
    times: Optional[str] = 'auto'
    average: Optional[float] = None

    @property
    def get_times(self):
        s = (self.times or '').strip().lower()
        if s == 'peak':
            return 'peak'
        if s == 'auto':
            return 'auto'
        try:
            numbers = [float(x) for x in re.findall(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)', self.times)]
        except Exception:
            return 'auto'
        filtered = [x for x in numbers if self.tmin <= x <= self.tmax]
        return filtered if filtered else 'auto'

@dataclass
class EvokedJointParamsDTO(EvokedParamsDTO,EvokedTopoParamsDTO):
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
