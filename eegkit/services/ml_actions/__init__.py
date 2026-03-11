"""AI actions registry and module loader.

Defines the shared `ai_registry` and `@register_ai` decorator. Submodules in
this package register concrete actions (list models, build dataset, train,
predict). The `EEGAIService` imports `ai_registry` and passes it to the base
service so the controller/UI can discover available actions and params.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type

# Shared registry: name -> {"params": DTO class | None, "function": unbound callable}
ml_registry: Dict[str, Dict[str, Any]] = {}


def register_ml(name: str, dto_cls: Optional[Type[Any]], category: str = "deep"):
    """Register an machine learning action with params DTO, handler function, and category."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        ml_registry[name] = {"params": dto_cls, "function": func, "category": category}
        return func

    return decorator


# Import submodules to populate registry.
from . import build_epoch_feature_dataset  # noqa: E402,F401
from . import train_feature_model  # noqa: E402,F401
from . import test_feature_models  # noqa: E402,F401