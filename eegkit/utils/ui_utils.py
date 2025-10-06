from dataclasses import fields, MISSING
from typing import get_origin, get_args

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


def make_widget(value, field=None):
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

    # Prefer field.type when provided (covers Optional[...] cases)
    is_opt_float = is_opt_int = False
    if field is not None:
        ftype = getattr(field, 'type', None)
        origin = get_origin(ftype)
        args = get_args(ftype) if origin is not None else ()
        base = origin or ftype
        is_opt_float = (origin is not None and float in args)
        is_opt_int = (origin is not None and int in args)

        # Optional numeric widgets: render as Text with placeholder showing current value
        if is_opt_float:
            ph = "" if value is None else str(value)
            return widgets.Text(value="", placeholder=ph, layout=widgets.Layout(width='150px'))
        if is_opt_int:
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


def read_widget(widget, default, wrap_list=False, field=None):
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
    # Prefer field.type over default's type when available (handles Optional[...] types)
    if field is not None:
        ftype = getattr(field, 'type', None)
        origin = get_origin(ftype)
        args = get_args(ftype) if origin is not None else ()

        def _is_optional_of(py_t):
            return (origin is None and ftype is py_t) or (origin is not None and py_t in args)

        # Optional numeric fields: empty input -> None (placeholder does not set value)
        if _is_optional_of(float):
            try:
                if isinstance(val, str) and val.strip() == "":
                    return None
                return float(val)
            except Exception:
                return None
        if _is_optional_of(int):
            try:
                if isinstance(val, str) and val.strip() == "":
                    return None
                return int(float(val))
            except Exception:
                return None
        if _is_optional_of(bool):
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
    """
    Return dataclass fields sorted by custom priority:
    int/float -> dropdown(list/enum) -> str -> bool -> others.
    """

    def priority(f):
        t = f.type
        origin = get_origin(t)
        args = get_args(t) if origin is not None else ()
        base = origin or t

        def _is_optional_of(py_t):
            return (origin is None and base is py_t) or (origin is not None and py_t in args)

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

    return sorted(fields(dto_cls), key=priority)


