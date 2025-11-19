"""Base task and representation DTOs plus shared helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields, asdict
from typing import Optional, List, Tuple, ClassVar, Dict, Set

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

__all__ = [
    "NumberRange",
    "make_hashable",
    "BaseTaskDTO",
    "TaskDTO",
    "SubjectFilterDTO",
    "ReprMixin",
]
