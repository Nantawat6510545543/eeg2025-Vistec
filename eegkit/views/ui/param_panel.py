"""Parameter panel builder for EEG GUI.

Responsible for constructing parameter widgets for each action (group/key),
organizing them by owner class, wiring observers to UIParamCache, and
collecting DTO values for execution.
"""
from __future__ import annotations

from dataclasses import fields
from typing import Any, Dict, List, Tuple

import ipywidgets as widgets

from ..ui.widgets import make_widget, ordered_fields, read_widget
from ..ui_support.cache import UIParamCache
from ..ui_support.grouping import derive_param_groups


class ParamPanel:
    """Builder for parameter input widgets grouped by owning classes."""

    def __init__(self, controller, ui_param_cache: UIParamCache) -> None:
        """Store controller, cache, and initialize layout registries."""
        self.controller = controller
        self.ui_param_cache = ui_param_cache
        self.param_layouts: Dict[Tuple[str, str], List[widgets.Widget]] = {}
        self.param_widgets: Dict[Tuple[str, str], Dict[str, widgets.Widget | dict]] = {}
        self.param_group_meta: Dict[Tuple[str, str], List[dict]] = {}
        self._field_owner: Dict[Tuple[str, str], Dict[str, type]] = {}
        self._wired_layouts: set[Tuple[str, str]] = set()

    def ensure_layout(self, group: str, key: str, params_cls: Any | None) -> None:
        """Build and cache the layout/widgets for a (group, key) action if missing."""
        layout_key = (group, key)
        if layout_key in self.param_layouts:
            return

        widgets_dict: Dict[str, widgets.Widget | dict] = {}
        group_meta: List[dict] = []

        if params_cls:
            params_obj = params_cls() if callable(params_cls) else params_cls
            groups = derive_param_groups(params_obj)
            for g in groups:
                owner = g["owner"]
                title = g["title"]
                field_names = g["field_names"]
                rows_for_owner: List[widgets.Widget] = []
                pair_buf: List[widgets.Widget] = []
                ordered_owner_fields = [f.name for f in ordered_fields(owner)]
                ordered_names = [n for n in ordered_owner_fields if n in field_names] or field_names
                for name in ordered_names:
                    if not hasattr(params_obj, name):
                        continue
                    val = getattr(params_obj, name)
                    w = make_widget(val, field=next(f for f in fields(params_obj) if f.name == name))
                    widgets_dict[name] = w
                    row = widgets.HBox([
                        widgets.Label(value=f"{name}:", layout=widgets.Layout(width='200px')),
                        w
                    ])
                    if isinstance(val, str):
                        if pair_buf:
                            rows_for_owner.append(widgets.HBox(pair_buf))
                            pair_buf = []
                        rows_for_owner.append(row)
                    else:
                        pair_buf.append(row)
                        if len(pair_buf) == 2:
                            rows_for_owner.append(widgets.HBox(pair_buf))
                            pair_buf = []
                if pair_buf:
                    rows_for_owner.append(widgets.HBox(pair_buf))
                group_meta.append({
                    "owner": owner,
                    "title": title,
                    "field_names": field_names,
                    "rows": rows_for_owner,
                })

        flat_rows: List[widgets.Widget] = []
        for gm in group_meta:
            flat_rows.extend(gm["rows"])
        self.param_layouts[layout_key] = flat_rows
        self.param_widgets[layout_key] = widgets_dict
        self.param_group_meta[layout_key] = group_meta
        # field -> owner
        owner_map: Dict[str, type] = {}
        for gm in group_meta:
            for n in gm["field_names"]:
                owner_map[n] = gm["owner"]
        self._field_owner[layout_key] = owner_map
        # wire observers once
        self._wire_param_observers(layout_key, group_meta, widgets_dict)

    def _wire_param_observers(self, layout_key, group_meta, widgets_dict):
        if layout_key in self._wired_layouts:
            return
        owner_map = self._field_owner.get(layout_key, {})
        for name, w in widgets_dict.items():
            owner = owner_map.get(name)
            if owner is None:
                continue

            def _make_handler(_name=name, _owner=owner, _w=w):
                def _on_change(change):
                    if change.get('name') != 'value':
                        return
                    val = getattr(_w, 'value', None)
                    self.ui_param_cache.set_value(_owner, _name, val)

                return _on_change

            if isinstance(w, dict):
                for part in ('min', 'max'):
                    if part in w and hasattr(w[part], 'observe'):
                        w[part].observe(_make_handler(), names='value')
            elif hasattr(w, 'observe'):
                w.observe(_make_handler(), names='value')
        self._wired_layouts.add(layout_key)

    def collect_params(self, group: str, key: str, spec: dict, param_widgets: Dict[Tuple[str, str], Dict[str, Any]]):
        """Collect current widget values and construct the params DTO instance."""
        params_cls = spec.get("params")
        if not params_cls:
            return None
        defaults = params_cls()
        widgets_map = param_widgets.get((group, key), {})
        values: Dict[str, Any] = {}
        for f in fields(defaults):
            values[f.name] = read_widget(
                widgets_map.get(f.name), getattr(defaults, f.name), wrap_list=False, field=f
            )
        return params_cls(**values)
