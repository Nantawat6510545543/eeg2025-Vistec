"""Filtering, epoching, PSD, evoked, and table parameter DTOs."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, List, ClassVar, Set, Dict, Union

from .base import ReprMixin


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

    showbad: bool = False
    clean_flatline_sec: Optional[float] = 5.0
    clean_hf_noise_sd_max: Optional[float] = 4.0
    clean_corr_min: Optional[float] = 0.8

    clean_asr_max_std: Optional[float] = 20.0
    clean_asr_remove_only: bool = False

    clean_power_min_sd: Optional[float] = -100.0
    clean_power_max_sd: Optional[float] = 7.0
    clean_max_outbound_pct: Optional[float] = 25.0
    clean_window_sec: Optional[float] = 0.5

    _exclude_str_fields: ClassVar[Set[str]] = {
        "combine_channels",
        "showbad",
    }

    @property
    def filter_key(self) -> Dict[str, float]:
        """Return cache key for prefilter stage (band-pass, resample, notch)."""
        return {
            "l_freq": self.l_freq,
            "h_freq": self.h_freq,
            "notch": self.notch,
            "resample_fs": self.resample_fs,
        }

    @property
    def cleaning_key(self) -> Dict[str, float | bool]:
        """Return cache key for cleaning/marking stage (bad channels/windows)."""
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
        """Return cache key for epoching stage (includes tmin/tmax)."""
        return {**self.cleaning_key, "tmin": self.tmin, "tmax": self.tmax}

    @property
    def evoked_key(self):
        """Return cache key for evoked stage (epochs_key + stimulus)."""
        return {**self.epochs_key, "stimulus": self.stimulus}


@dataclass
class PSDParamsDTO(FilterParamsDTO):
    """Power Spectral Density parameters (frequency range and display flags)."""

    fmin: Optional[float] = None
    fmax: Optional[float] = None
    average: bool = True
    dB: bool = True
    spatial_colors: bool = True
    _exclude_str_fields: ClassVar[Set[str]] = {"average", "dB", "spatial_colors"}

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
    error_band: List[Optional[str]] = field(default_factory=lambda: [None, "sem", "std"]) 
    scale_mode: List[str] = field(default_factory=lambda: ["per-plot", "uniform-grid"])
    _exclude_str_fields: ClassVar[Set[str]] = {"spatial_colors", "gfp", "average_line", "error_band", "scale_mode"}


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

__all__ = [
    "FilterParamsDTO",
    "EpochParamsDTO",
    "PSDParamsDTO",
    "EpochPSDParamsDTO",
    "EvokedParamsDTO",
    "EvokedTopoParamsDTO",
    "EvokedJointParamsDTO",
    "TimeDomainParamsDTO",
    "TableInfoDTO",
]
