"""DL actions registry and module loader.

Defines the shared `dl_registry` and `@register_dl` decorator. Submodules in
this package register concrete actions (build dataset, train EEGNet).
The `EEGDLService` imports `dl_registry` and passes it to the base service
so the controller/UI can discover available actions and params.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type

# Shared registry: name -> {"params": DTO class | None, "function": unbound callable}
dl_registry: Dict[str, Dict[str, Any]] = {}


def register_dl(name: str, dto_cls: Optional[Type[Any]], category: str = "deep"):
    """Register a deep learning action with params DTO, handler function, and category."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        dl_registry[name] = {"params": dto_cls, "function": func, "category": category}
        return func

    return decorator


# Import submodules to populate registry.
from . import build_dataset  # noqa: E402,F401
from . import train_eegnet   # noqa: E402,F401
