from dataclasses import fields, MISSING

import ipywidgets as widgets

from ..models import BaseTaskDTO


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


def make_widget(value):
    """Create a widget appropriate for the given default value type."""
    if isinstance(value, tuple) and len(value) == 2 and all(isinstance(x, (int, float)) for x in value):
        vmin, vmax = float(value[0]), float(value[1])
        return {
            "min": widgets.FloatText(value=vmin, layout=widgets.Layout(width='100px')),
            "max": widgets.FloatText(value=vmax, layout=widgets.Layout(width='100px')),
        }
    if isinstance(value, bool):
        return widgets.Checkbox(value=value, layout=widgets.Layout(width='150px'))
    if isinstance(value, int):
        return widgets.IntText(value=value, layout=widgets.Layout(width='150px'))
    if isinstance(value, float):
        return widgets.FloatText(value=value, layout=widgets.Layout(width='150px'))
    if isinstance(value, list):
        return widgets.Dropdown(options=value, layout=widgets.Layout(width='200px'))
    return widgets.Text(value="" if value is None else str(value), layout=widgets.Layout(width='500'))


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


def read_widget(widget, default, wrap_list=False):
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
    if isinstance(default, bool):
        return bool(val)
    if isinstance(default, int):
        return int(val)
    if isinstance(default, float):
        return float(val)
    return val
