"""Reusable ipywidgets helpers for building EEG parameter forms."""
from __future__ import annotations

from dataclasses import fields, MISSING
from typing import get_origin, get_args, get_type_hints, Annotated

import ipywidgets as widgets


def is_subject_schema(schema_dto):
    """Return True if the DTO schema includes a 'subject' field (single-subject mode)."""
    return any(f.name == "subject" for f in fields(schema_dto))


def field_default(f):
    """Resolve the default value for a dataclass field, honoring default_factory."""
    if f.default is not MISSING:
        return f.default
    if getattr(f, "default_factory", MISSING) is not MISSING:
        return f.default_factory()
    return None


def make_widget(value, field=None, owner_cls=None):
    """Create a widget appropriate for the given field/value type.

    - Uses dataclass field.type when available to correctly handle Optional[float]/Optional[int].
    - Standardizes control widths for a more consistent UI.
    """
    if isinstance(value, tuple) and len(value) == 2 and all(isinstance(x, (int, float)) for x in value):
        vmin, vmax = float(value[0]), float(value[1])
        return {
            "min": widgets.FloatText(value=vmin, layout=widgets.Layout(width='100px')),
            "max": widgets.FloatText(value=vmax, layout=widgets.Layout(width='100px')),
        }

    # Prefer resolved annotations when provided (covers Optional[...] and Annotated)
    if field is not None:
        ftype = getattr(field, 'type', None)
        try:
            hints = get_type_hints(owner_cls, include_extras=True) if owner_cls is not None else {}
        except Exception:
            hints = {}
        ann = hints.get(getattr(field, 'name', None), ftype)
        t_origin = get_origin(ann)
        if t_origin is Annotated:
            inner = get_args(ann)
            ann = inner[0] if inner else ann
            t_origin = get_origin(ann)
        origin = t_origin
        args = get_args(ann) if origin is not None else ()
        base = origin or ann

        args_set = set(args) if args else set()
        is_optional = any(a is type(None) for a in args_set)
        has_float = (float in args_set) or (base is float)
        has_int = (int in args_set) or (base is int)

        # Optional numeric widgets: render as Text with placeholder showing current value
        if is_optional and has_float:
            ph = "" if value is None else str(value)
            return widgets.Text(value="", placeholder=ph, layout=widgets.Layout(width='150px'))
        if is_optional and has_int:
            ph = "" if value is None else str(int(value))
            return widgets.Text(value="", placeholder=ph, layout=widgets.Layout(width='150px'))

    # Non-optional standard widgets
    if isinstance(value, bool):
        return widgets.Checkbox(value=value, layout=widgets.Layout(width='150px'))
    if isinstance(value, int):
        return widgets.IntText(value=value, layout=widgets.Layout(width='150px'))
    if isinstance(value, float):
        return widgets.FloatText(value=value, layout=widgets.Layout(width='150px'))
    if isinstance(value, list):
        return widgets.Dropdown(options=value, layout=widgets.Layout(width='150px'))

    # Fallback to a compact Text widget
    return widgets.Text(value="" if value is None else str(value), layout=widgets.Layout(width='500px'))


def make_range_widget(default_value):
    """Create a (min,max) text widget pair supporting open-ended bounds (None).

    default_value is expected to be a tuple (lower, upper), each possibly None.
    """
    if isinstance(default_value, (tuple, list)) and len(default_value) == 2:
        lower, upper = default_value
    else:
        lower, upper = (None, None)
    return {
        "min": widgets.Text(value='' if lower is None else str(lower), placeholder='min',
                            layout=widgets.Layout(width='100px')),
        "max": widgets.Text(value='' if upper is None else str(upper), placeholder='max',
                            layout=widgets.Layout(width='100px')),
    }


def read_widget(widget, default, wrap_list=False, field=None, owner_cls=None):
    """Read current widget value(s) and coerce to the expected type/shape.

    - For range dicts, returns a (lower, upper) with blanks mapped to None.
    - For primitives, casts to bool/int/float when default indicates type.
    - If wrap_list=True and default is a list, wraps the value in a list.
    """
    if isinstance(widget, dict):
        vmin = widget["min"].value
        vmax = widget["max"].value
        lower = None if vmin in (None, "") else float(vmin)
        upper = None if vmax in (None, "") else float(vmax)
        if isinstance(default, (tuple, list)) and len(default) == 2:
            if lower is None and upper is None:
                return None, None
        return lower, upper
    val = getattr(widget, "value", None)
    if wrap_list and isinstance(default, list):
        return [val]
    # Prefer resolved annotations when available (handles Optional[...] and Annotated)
    if field is not None:
        ftype = getattr(field, 'type', None)
        try:
            hints = get_type_hints(owner_cls, include_extras=True) if owner_cls is not None else {}
        except Exception:
            hints = {}
        ann = hints.get(getattr(field, 'name', None), ftype)
        t_origin = get_origin(ann)
        if t_origin is Annotated:
            inner = get_args(ann)
            ann = inner[0] if inner else ann
            t_origin = get_origin(ann)

        origin = t_origin
        args = get_args(ann) if origin is not None else ()
        args_set = set(args) if args else set()
        is_optional = any(a is type(None) for a in args_set)

        # Optional numeric fields: empty input -> None (placeholder does not set value)
        if (float in args_set or ann is float) and (is_optional or ann is float):
            try:
                if isinstance(val, str) and val.strip() == "":
                    return None
                return float(val)
            except Exception:
                return None
        if (int in args_set or ann is int) and (is_optional or ann is int):
            try:
                if isinstance(val, str) and val.strip() == "":
                    return None
                return int(float(val))
            except Exception:
                return None
        if (bool in args_set or ann is bool):
            try:
                return bool(val)
            except Exception:
                return False

    # Fallback: infer from default's runtime type
    if isinstance(default, bool):
        return bool(val)
    if isinstance(default, int):
        return int(val)
    if isinstance(default, float):
        return float(val)
    return val


def ordered_fields(dto_cls):
    """Return dataclass fields sorted by custom priority.

    int/float -> dropdown(list/enum) -> str -> bool -> others.
    """

    def priority(f):
        # Resolve postponed annotations to real types when possible
        try:
            hints = get_type_hints(dto_cls, include_extras=True)
        except Exception:
            hints = {}
        t = hints.get(f.name, f.type)

        # Unwrap Annotated[...] to its underlying type
        t_origin = get_origin(t)
        if t_origin is Annotated:
            t = get_args(t)[0] if get_args(t) else t
            t_origin = get_origin(t)

        origin = t_origin
        args = get_args(t) if origin is not None else ()
        base = origin or t

        def _is_optional_of(py_t):
            # Optional[T] is Union[T, NoneType] in modern typing; also allow bare T
            if origin is None:
                return base is py_t
            try:
                return any((a is py_t) for a in args)
            except Exception:
                return False

        # 1. Numeric first (including Optional[int/float])
        if _is_optional_of(int) or _is_optional_of(float):
            return 0

        # 2. Dropdown-like (default is list or type is list/Enum)
        default = None
        if f.default is not MISSING:
            default = f.default
        elif getattr(f, "default_factory", MISSING) is not MISSING:
            try:
                default = f.default_factory()
            except Exception:
                pass
        if isinstance(default, list) or base is list:
            return 1

        # 3. Strings
        if base is str:
            return 2

        # 4. Booleans
        if _is_optional_of(bool):
            return 3

        # 5. Everything else last
        return 4

    from dataclasses import fields as _fields
    return sorted(_fields(dto_cls), key=priority)


__all__ = [
    "is_subject_schema",
    "field_default",
    "make_widget",
    "make_range_widget",
    "read_widget",
    "ordered_fields",
]
