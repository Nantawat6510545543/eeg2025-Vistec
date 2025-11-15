"""Plot sub-services registry and module loader.

This package hosts small, focused plot implementations that register themselves
into a shared registry via the `@register_plot` decorator. The top-level
`EEGVisualization` service imports `plot_registry` from here and passes it to
`BaseService` so the controller/UI can discover available plots and params.
"""

from __future__ import annotations

from typing import Dict, Any, Callable, Type

# Shared registry: name -> {"params": DTO class, "function": unbound method}
plot_registry: Dict[str, Dict[str, Any]] = {}


def register_plot(name: str, dto_cls: Type[Any]) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to register a plotting function under a human-readable name.

    The decorated callable must accept `(self, task_dto, params)` and will be
    bound to the concrete `BaseService` subclass instance at runtime.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        plot_registry[name] = {"params": dto_cls, "function": func}
        return func

    return decorator


# Import all plot modules so they register themselves on import
from . import sensors  # noqa: E402,F401
from . import time_domain  # noqa: E402,F401
from . import frequency  # noqa: E402,F401
from . import epochs  # noqa: E402,F401
from . import evoked  # noqa: E402,F401
from . import snr  # noqa: E402,F401
