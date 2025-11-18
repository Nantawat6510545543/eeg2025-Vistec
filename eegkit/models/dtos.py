"""Dataclass DTO definitions for tasks, filtering, epochs, PSD, evoked and AI params."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields, asdict
from typing import Optional, List, Tuple, ClassVar, Dict, Set, Union

NumberRange = Tuple[Optional[float], Optional[float]]


def make_hashable(value):
    """Return hashable version of nested value (lists->tuples, sets->frozensets)."""
    if isinstance(value, list):
        return tuple(make_hashable(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in value.items()))
    if isinstance(value, set):
        return frozenset(make_hashable(v) for v in value)
    return value


@dataclass
class BaseTaskDTO:
    """Base task selector with common hashing behavior for DTOs."""

    task: str
    ui_name: ClassVar[str] = "Base"
    ui_value: ClassVar[object] = None

    def __hash__(self):
        """Hash by tuple of field values (normalized to hashable forms)."""
        return hash(tuple(make_hashable(getattr(self, f.name)) for f in fields(self)))


@dataclass
class TaskDTO(BaseTaskDTO):
    """Concrete single-subject task selector (subject, task, optional run)."""

    task: str
    subject: str
    run: Optional[str] = None

    ui_name: ClassVar[str] = "Single subject"
    ui_value: ClassVar[object] = None

    __hash__ = BaseTaskDTO.__hash__

    def __repr__(self) -> str:
        """Human-readable summary of subject/task/run."""
        return f"subject = {self.subject}, task = {self.task}, run = {self.run}"


@dataclass
class SubjectFilterDTO(BaseTaskDTO):
    """Cohort selector with demographic/behavioral filters and options."""

    task: str
    subject_limit: Optional[int] = None
    per_subject: bool = False
    sex: List[Optional[str]] = field(default_factory=lambda: [None, "M", "F"])
    age_range: Optional[NumberRange] = (None, None)
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
        """Compact description of active filters (for UI/debug)."""
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
class ReprMixin:
    """Reusable mixin for clean __repr__ across dataclasses."""

    _exclude_str_fields: ClassVar[Set[str]] = set()

    def __str__(self) -> str:
        """Render non-empty fields as key=value pairs."""
        data = asdict(self)
        parts = []
        for k, v in data.items():
            if k in self._exclude_str_fields:
                continue
            if v is None:
                continue
            if isinstance(v, str) and not v.strip():
                continue
            if isinstance(v, bool) and not v:
                continue
            parts.append(f"{k}={v}")
        return ', '.join(parts)

    def __init_subclass__(cls, **kwargs):
        """When subclassing, merge exclusions from all bases."""
        super().__init_subclass__(**kwargs)
        merged = set()
        for base in cls.__mro__:
            if hasattr(base, "_exclude_str_fields"):
                merged |= getattr(base, "_exclude_str_fields")
        cls._exclude_str_fields = merged


@dataclass
class FilterParamsDTO(ReprMixin):
    """Parameters controlling filtering, channel selection and cleaning."""

    l_freq: float = 4.0
    h_freq: float = 30.0
    notch: Optional[float] = 60.0
    resample_fs: float = 100.0
    channels: str = "69-76,81-83,88,89"
    combine_channels: bool = False

    uv_min: Optional[float] = -100.0
    uv_max: Optional[float] = 100.0

    # Remove bad channels
    showbad: bool = False  # if True, plot bad channels found
    clean_flatline_sec: Optional[float] = 5.0
    clean_hf_noise_sd_max: Optional[float] = 4.0
    clean_corr_min: Optional[float] = 0.8  # min acceptable absolute correlation to aggregate

    # ASR bad subspace correction/removal (requires optional asrpy; otherwise skipped)
    clean_asr_max_std: Optional[float] = 20.0  # max acceptable 0.5s window std dev (equiv.)
    clean_asr_remove_only: bool = False  # if True, only annotate/remove bad periods, no reconstruction

    # Additional removal of bad data periods
    clean_power_min_sd: Optional[float] = -100.0
    clean_power_max_sd: Optional[float] = 7.0
    clean_max_outbound_pct: Optional[float] = 25.0  # percentage of channels
    clean_window_sec: Optional[float] = 0.5  # analysis window size (s)

    _exclude_str_fields: ClassVar[Set[str]] = {
        "combine_channels",
        "showbad",
    }

    @property
    def filter_key(self) -> Dict[str, float]:
        """Cache key for prefilter-only stage (band-pass, resample, notch).

        This intentionally excludes any cleaning/marking options so that the
        heavy prefilter result can be reused regardless of how we later mark
        bad channels or bad time windows.
        """
        return {
            "l_freq": self.l_freq,
            "h_freq": self.h_freq,
            "notch": self.notch,  # may be None
            "resample_fs": self.resample_fs,
        }

    @property
    def cleaning_key(self) -> Dict[str, float | bool]:
        """Cache key for cleaning/marking stage (bad channels, bad periods).

        Include only non-None thresholds; booleans as-is. This prevents None from
        being coerced and allows Optional fields to disable steps without breaking the key.
        """
        key = {**self.filter_key}

        def add(name, val):
            if val is not None:
                key[name] = val

        add("clean_flatline_sec", self.clean_flatline_sec)
        add("clean_hf_noise_sd_max", self.clean_hf_noise_sd_max)
        add("clean_corr_min", self.clean_corr_min)
        add("clean_asr_max_std", self.clean_asr_max_std)

        if self.clean_asr_remove_only:
            key["clean_asr_remove_only"] = True

        add("clean_power_min_sd", self.clean_power_min_sd)
        add("clean_power_max_sd", self.clean_power_max_sd)
        add("clean_max_outbound_pct", self.clean_max_outbound_pct)
        add("clean_window_sec", self.clean_window_sec)

        return key

    @property
    def channels_list(self):
        """Parse channels string into unique list like ['E1','E2',...]."""
        if not self.channels or not self.channels.strip():
            return [f"E{i}" for i in range(1, 129)]

        number_or_range = re.compile(r'^(\d+)(?:\s*-\s*(\d+))?$')

        raw_tokens = re.split(r'[\s,]+', self.channels.strip())

        seen_channel_names = set()
        parsed_channel_names: list[str] = []

        def add_channel_name(channel_name: str) -> None:
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
    """Epoch extraction parameters extending filtering/cleaning options."""

    tmin: float = -2.0
    tmax: float = 0.0
    stimulus: List[Optional[str]] = field(default_factory=lambda: [None])
    only_labels: ClassVar[bool] = False

    _exclude_str_fields: ClassVar[Set[str]] = {"stimulus"}

    @property
    def epochs_key(self) -> Dict[str, float]:
        """Cache key for epoching stage (includes tmin/tmax)."""
        key = {
            **self.cleaning_key,
            "tmin": self.tmin,
            "tmax": self.tmax,
        }
        return key

    @property
    def evoked_key(self):
        """Cache key for evoked stage (epochs_key + stimulus)."""
        key = {
            **self.epochs_key,
            "stimulus": self.stimulus,
        }
        return key


@dataclass
class PSDParamsDTO(FilterParamsDTO):
    """Power Spectral Density parameters (frequency range and display flags)."""

    fmin: Optional[float] = None
    fmax: Optional[float] = None
    average: bool = True
    dB: bool = True
    spatial_colors: bool = True

    _exclude_str_fields: ClassVar[Set[str]] = {
        "average", "dB", "spatial_colors"
    }

    def __post_init__(self):
        """Set fmin/fmax to filter band if not provided."""
        if self.fmin is None:
            self.fmin = self.l_freq
        if self.fmax is None:
            self.fmax = self.h_freq


@dataclass
class EpochPSDParamsDTO(PSDParamsDTO, EpochParamsDTO):
    """PSD parameters combined with epoching options."""

    pass


@dataclass
class EvokedParamsDTO(EpochParamsDTO):
    """Parameters controlling evoked plotting/averaging options."""

    spatial_colors: bool = True
    gfp: List[Union[str, bool]] = field(default_factory=lambda: [False, True, "only"])  # True/"only"
    average_line: bool = True
    scale_mode: List[str] = field(default_factory=lambda: ["per-plot", "uniform-grid"])

    _exclude_str_fields: ClassVar[Set[str]] = {
        "spatial_colors", "gfp", "average_line", "scale_mode"
    }


@dataclass
class EvokedTopoParamsDTO(EpochParamsDTO):
    """Parameters for topomap visualization with selected times."""

    times: str = 'auto'

    @property
    def get_times(self):
        """Return 'peak'/'auto' or filtered numeric times within [tmin, tmax]."""
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
class EvokedJointParamsDTO(EvokedParamsDTO, EvokedTopoParamsDTO):
    """Combined params for joint evoked/time-topo plots."""

    pass


@dataclass
class TimeDomainParamsDTO(FilterParamsDTO):
    """Parameters for raw/time-domain preview plots."""

    duration: float = 10.0
    start: float = 0.0
    n_channels: int = 10


@dataclass
class TableInfoDTO(FilterParamsDTO):
    """Parameters selecting data table type and size for previews."""

    table_type: List[str] = field(default_factory=lambda: ["events", "channels", "electrodes"])
    rows: int = 10


# ---- AI DTOs ----
@dataclass
class AIBaseDTO(EpochParamsDTO):
    """Base AI params including model selection list."""

    model: List[Optional[str]] = field(default_factory=lambda: [None])


@dataclass
class AITrainParamsDTO(AIBaseDTO):
    """Training hyperparameters and target selection for simple trainers."""

    batch_size: int = 32
    epochs_n: int = 1
    lr: float = 0.001
    device: List[str] = field(default_factory=lambda: ["auto", "cpu", "cuda"])
    target: List[str] = field(default_factory=lambda: ["stimulus"])


@dataclass
class AIPredictParamsDTO(AIBaseDTO):
    """Prediction params for future checkpoint-based inference."""

    checkpoint_path: Optional[str] = None
