"""AI actions registry and module loader.

Defines the shared `ai_registry` and `@register_ai` decorator. Submodules in
this package register concrete actions (list models, build dataset, train,
predict). The `EEGAIService` imports `ai_registry` and passes it to the base
service so the controller/UI can discover available actions and params.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type

from ...models.dtos import AIBaseDTO

# Shared registry: name -> {"params": DTO class | None, "function": unbound callable}
ai_registry: Dict[str, Dict[str, Any]] = {}


def register_ai(name: str, dto_cls: Optional[Type[AIBaseDTO]]):
    """Register an AI action with its params DTO class and handler function."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        ai_registry[name] = {"params": dto_cls, "function": func}
        return func

    return decorator


# Import submodules to populate registry on package import
from . import list_models  # noqa: E402,F401
from . import dataset  # noqa: E402,F401
from . import train  # noqa: E402,F401
from . import predict  # noqa: E402,F401
