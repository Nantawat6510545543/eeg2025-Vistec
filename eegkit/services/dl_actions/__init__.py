"""AI actions registry and module loader.

Defines the shared `ai_registry` and `@register_ai` decorator. Submodules in
this package register concrete actions (list models, build dataset, train,
predict). The `EEGAIService` imports `ai_registry` and passes it to the base
service so the controller/UI can discover available actions and params.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type

from ...models.dtos import DLTrainParamsDTO

# Shared registry: name -> {"params": DTO class | None, "function": unbound callable}
dl_registry: Dict[str, Dict[str, Any]] = {}


def register_dl(name: str, dto_cls: Optional[Type[DLTrainParamsDTO]], category: str = "deep"):
    """Register an deep learning action with params DTO, handler function, and category."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        dl_registry[name] = {"params": dto_cls, "function": func, "category": category}
        return func

    return decorator