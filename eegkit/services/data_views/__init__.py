"""Data view sub-services registry and module loader.

Holds small tabular view implementations registered via `@register_data`.
`EEGDataService` uses the shared `data_registry` from here.
"""

from __future__ import annotations

from typing import Dict, Any, Callable, Type

data_registry: Dict[str, Dict[str, Any]] = {}


def register_data(name: str, dto_cls: Type[Any] | None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        data_registry[name] = {"params": dto_cls, "function": func}
        return func

    return decorator


# Import submodules to populate registry
from . import eeg_table  # noqa: F401
from . import epochs_table  # noqa: F401
from . import annotations  # noqa: F401
from . import metadata  # noqa: F401
