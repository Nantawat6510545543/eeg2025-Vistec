"""Group parameter fields by defining class (no ipywidgets dependency)."""
from __future__ import annotations

from dataclasses import is_dataclass, fields
from typing import Any, Dict, List, Tuple, Type


def _camel_to_title(name: str) -> str:
    import re
    # strip common suffixes
    for suf in ("ParamsDTO", "DTO", "Params"):
        if name.endswith(suf):
            name = name[: -len(suf)]
            break
    # split CamelCase
    parts = re.findall(r"[A-Za-z][^A-Z]*", name) or [name]
    return " ".join(parts)


TITLE_OVERRIDES: Dict[str, str] = {
    # class name -> label shown in UI
    "FilterParamsDTO": "Filtering & Cleaning",
    "EpochParamsDTO": "Epochs",
    "PSDParamsDTO": "PSD",
    "EvokedParamsDTO": "Evoked Display",
    "EvokedTopoParamsDTO": "Topomap",
    "EvokedJointParamsDTO": "Joint Plot",
    "TimeDomainParamsDTO": "Time Domain",
    "TableInfoDTO": "Tables",
    "DLTrainParamsDTO": "Training",
    "EEGNetBinaryTrainParamsDTO": "Training",
    "EEGNetRegTrainParamsDTO": "Training",
    "DLTrainRuntimeParamsDTO": "Runtime",
    "DLTrainSplitParamsDTO": "Split",
    "DLTrainIOParamsDTO": "I/O",
    "DLTrainSchedulerParamsDTO": "Scheduler",
    "DLTrainBalanceParamsDTO": "Balance",
    "DLTrainLossParamsDTO": "Loss",
    "DLTrainRegParamsDTO": "Regression",
    "MLTrainParamsDTO": "Training",
}


def _dataclass_field_names(cls_or_obj: Any) -> List[str]:
    if not is_dataclass(cls_or_obj):
        if isinstance(cls_or_obj, type) and hasattr(cls_or_obj, "__dataclass_fields__"):
            return [f.name for f in fields(cls_or_obj)]
        return []
    return [f.name for f in fields(cls_or_obj)]


def derive_param_groups(cls_or_instance: Any) -> List[Dict[str, Any]]:
    """Return ordered groups derived from MRO.

    Each group is a dict: {"owner": type, "title": str, "field_names": [str, ...]}
    """
    params_cls: Type[Any] = cls_or_instance if isinstance(cls_or_instance, type) else type(cls_or_instance)

    all_field_names = set(_dataclass_field_names(cls_or_instance))
    if not all_field_names:
        return []

    mro = list(params_cls.__mro__)

    def _is_valid_owner(t: Type[Any]) -> bool:
        return t is not object and hasattr(t, "__dict__")

    assigned: Dict[str, Type[Any]] = {}
    groups_rev: List[Tuple[Type[Any], List[str]]] = []

    for owner in reversed([t for t in mro if _is_valid_owner(t)]):
        ann = getattr(owner, "__annotations__", {}) or {}
        owned = [n for n in ann.keys() if n in all_field_names and n not in assigned]
        if not owned:
            continue
        for n in owned:
            assigned[n] = owner
        groups_rev.append((owner, owned))

    result: List[Dict[str, Any]] = []
    for owner, names in groups_rev:
        title = TITLE_OVERRIDES.get(owner.__name__, _camel_to_title(owner.__name__))
        result.append({
            "owner": owner,
            "title": title,
            "field_names": names,
        })
    return result


__all__ = ["derive_param_groups", "TITLE_OVERRIDES"]
