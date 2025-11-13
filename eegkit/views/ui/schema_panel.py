"""Schema panel builder for EEG GUI.

Responsible for constructing the TaskDTO / SubjectFilterDTO forms, wiring
subject changes to refresh available tasks, and exposing the built widgets
and layout rows to the orchestrator.
"""
from __future__ import annotations

from dataclasses import fields
from typing import Callable, Dict, List, Type

import ipywidgets as widgets

from ..ui.widgets import is_subject_schema, field_default, make_widget, make_range_widget
from ..ui_support.cache import UIParamCache  # only for typing reference if needed


class SchemaPanel:
    """Builds schema-specific forms and manages subject→task refresh.

    Parameters
    - controller: object providing list_subjects(), list_all_tasks(), list_tasks(subject)
    - schemas: list of dataclass types (e.g., [TaskDTO, SubjectFilterDTO])
    - on_subject_change: optional callback called when subject changes
    """

    def __init__(
        self,
        controller,
        schemas: List[Type],
        on_subject_change: Callable[[], None] | None = None,
    ) -> None:
        self.controller = controller
        self.schemas = list(schemas)
        self.on_subject_change = on_subject_change

        self.schema_layouts: Dict[Type, List[widgets.Widget]] = {}
        self.schema_widgets: Dict[Type, Dict[str, widgets.Widget | dict]] = {}

        self._build_all_schema_layouts()

    # public API --------------------------------------------------------------
    def get_layout_rows(self, schema_dto: Type) -> List[widgets.Widget]:
        return self.schema_layouts.get(schema_dto, [])

    def get_widgets_map(self, schema_dto: Type) -> Dict[str, widgets.Widget | dict]:
        return self.schema_widgets.get(schema_dto, {})

    def refresh_task_options(self, schema_dto: Type, init: bool = False) -> None:
        """Populate the task dropdown for the selected subject.
        Preserves selection on init when possible.
        """
        wmap = self.schema_widgets.get(schema_dto)
        if not wmap:
            return
        subj_w = wmap.get("subject")
        task_w = wmap.get("task")
        if subj_w is None or task_w is None:
            return
        subj = getattr(subj_w, "value", None)
        tasks = self.controller.list_tasks(subj) if subj else []
        opts = []
        for t, r in tasks:
            label = f"{t} (Run {r})" if r else f"{t}"
            opts.append((label, (t, r)))
        task_w.options = opts
        if not opts:
            task_w.value = None
            return
        if init:
            if task_w.value not in [v for _, v in opts]:
                task_w.value = opts[0][1]
        else:
            task_w.value = opts[0][1]

    # internal ----------------------------------------------------------------
    def _build_all_schema_layouts(self) -> None:
        all_subjects = self.controller.list_subjects()
        all_tasks = self.controller.list_all_tasks()
        for schema_dto in self.schemas:
            rows: List[widgets.Widget] = []
            wmap: Dict[str, widgets.Widget | dict] = {}
            subject_schema = is_subject_schema(schema_dto)

            schema_fields = list(fields(schema_dto))
            if subject_schema:
                schema_fields.sort(
                    key=lambda field: (0 if field.name == 'subject' else (1 if field.name == 'task' else 2))
                )

            for f in schema_fields:
                name = f.name
                if subject_schema and name == 'run':
                    continue

                label = widgets.Label(value=f"{name}:", layout=widgets.Layout(width='200px'))
                if subject_schema and name == "subject":
                    w = widgets.Dropdown(options=all_subjects, layout=widgets.Layout(width='220px'))
                    # Wire subject changes: refresh tasks and bubble up
                    def _handler(change, _schema=schema_dto):
                        if change.get('name') != 'value':
                            return
                        self.refresh_task_options(_schema, init=False)
                        if callable(self.on_subject_change):
                            self.on_subject_change()
                    w.observe(_handler, names="value")
                elif subject_schema and name == "task":
                    w = widgets.Dropdown(options=[], layout=widgets.Layout(width='220px'))
                elif (not subject_schema) and name == "task":
                    w = widgets.Dropdown(options=all_tasks, layout=widgets.Layout(width='220px'))
                elif (schema_dto.__name__ == 'SubjectFilterDTO') and name.endswith('_range'):
                    default_val = field_default(f)
                    w = make_range_widget(default_val)
                else:
                    default_val = field_default(f)
                    w = make_widget(default_val, field=f)

                wmap[name] = w
                if isinstance(w, dict):
                    pair = widgets.HBox([
                        widgets.Label("min", layout=widgets.Layout(width="40px")), w["min"],
                        widgets.Label("max", layout=widgets.Layout(width="40px")), w["max"],
                    ])
                    rows.append(widgets.HBox([label, pair]))
                else:
                    rows.append(widgets.HBox([label, w]))

            self.schema_layouts[schema_dto] = rows
            self.schema_widgets[schema_dto] = wmap

            if subject_schema:
                self.refresh_task_options(schema_dto, init=True)
