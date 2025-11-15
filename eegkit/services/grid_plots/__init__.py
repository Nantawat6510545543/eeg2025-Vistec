"""Grid plot sub-services registry and module loader.

Hosts small, focused grid plot implementations that register themselves via
`@register_grid_plot`. The `EEGGridVisualization` service imports the shared
`grid_plot_registry` from here and passes it to `BaseService`.
"""

from __future__ import annotations

from typing import Dict, Any, Callable, Type

grid_plot_registry: Dict[str, Dict[str, Any]] = {}


def register_grid_plot(name: str, dto_cls: Type[Any]) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        grid_plot_registry[name] = {"params": dto_cls, "function": func}
        return func

    return decorator


# Import submodules to populate registry
from . import psd_grid  # noqa: E402,F401
from . import snr_grid  # noqa: E402,F401
from . import evoked_grid  # noqa: E402,F401
