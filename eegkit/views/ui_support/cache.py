"""UI-only parameter cache (pure, no ipywidgets)."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Type


class UIParamCache:
    """Simple in-memory cache of UI parameter values keyed by owner class."""

    def __init__(self) -> None:
        """Initialize empty store mapping owner classes to field values."""
        self._store: Dict[Type[Any], Dict[str, Any]] = {}

    def clear(self, owner: Type[Any] | None = None) -> None:
        """Clear all cached values or those for a specific owner class."""
        if owner is None:
            self._store.clear()
        else:
            self._store.pop(owner, None)

    def set_value(self, owner: Type[Any], field_name: str, value: Any) -> None:
        """Store a value for a given owner class field name."""
        bucket = self._store.setdefault(owner, {})
        bucket[field_name] = value

    def get_value(self, owner: Type[Any], field_name: str) -> Any:
        """Return the cached value for owner.field_name or None if missing."""
        return self._store.get(owner, {}).get(field_name, None)

    def get_overlay(self, group_meta: List[Mapping[str, Any]]) -> Dict[str, Any]:
        """Compose a flat field overlay using base-to-derived order.

        group_meta: [{"owner": type, "title": str, "field_names": [..], ...}]
        Returns: { field_name: cached_value }
        """
        overlay: Dict[str, Any] = {}
        if not group_meta:
            return overlay
        for gm in group_meta:
            owner = gm.get("owner")
            field_names = gm.get("field_names", [])
            cached = self._store.get(owner)
            if not cached:
                continue
            for name in field_names:
                if name in cached:
                    overlay[name] = cached[name]
        return overlay


__all__ = ["UIParamCache"]
