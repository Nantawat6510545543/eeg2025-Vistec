"""EEG ipywidgets-based GUI

Provides a lightweight interface to:
- Choose input type (single subject or meta filter)
- Select mode (plot/data) and an action
- Adjust parameters and schedule a job via JobRunner

Notes:
- SubjectFilterDTO supports open-ended numeric ranges via min/max text boxes.
- If exposed in the form, the per-subject batch option runs the chosen action
    separately for each subject matched by the filter (subject limit respected).
"""
import json
from dataclasses import fields

import ipywidgets as widgets
import pandas as pd
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
from .job_runner import JobRunner
from pathlib import Path

from ..models import BaseTaskDTO, TaskDTO, SubjectFilterDTO
from ..utils import is_subject_schema, field_default, make_widget, make_range_widget, read_widget, ordered_fields
from ..utils.ui_param_groups import derive_param_groups

plt.ioff()


class EEGUI:
    """Interactive UI wrapper around EEGController.

    Builds dynamic forms from DTO definitions and available specs, and queues
    asynchronous jobs that render plots or export data. Designed for use in
    notebooks (ipywidgets).
    """

    def __init__(self, controller):
        """Initialize the GUI with a controller and build all widget layouts."""
        self.controller = controller
        self.specs = self.controller.get_specs()
        self.mode_info = getattr(self.controller, 'get_modes_info', lambda: {})()

        self.schemas = [TaskDTO, SubjectFilterDTO]
        self.schema_selector = widgets.ToggleButtons(
            options=[(getattr(c, "ui_name", c.__name__), c) for c in self.schemas],
            description="Input Type:", layout=widgets.Layout(width="auto")
        )
        self.schema_selector.value = self.schemas[0]

        self.mode_selector = widgets.ToggleButtons(options=list(self.specs.keys()), description="Mode:")
        self.mode_description = widgets.HTML(value="")
        self.action_selector = widgets.ToggleButtons(description="Action:")
        self.param_view_toggle = widgets.Checkbox(
            value=False,
            description="Show all groups",
            indent=False,
            layout=widgets.Layout(width="auto")
        )
        self.param_box = widgets.VBox()
        self.tmux_run_button = widgets.Button(description="Run on Tmux", button_style="success")
        self.inline_run_button = widgets.Button(description="Run Inline", button_style="success")
        self.output = widgets.Output()

        self.jobs_root = Path("jobs")
        self.jobs_root.mkdir(exist_ok=True, parents=True)

        self.schema_layouts = {}
        self.schema_widgets = {}
        self._build_all_schema_layouts()

        self.param_layouts = {}
        self.param_widgets = {}
        self.param_group_meta = {}
        self._build_all_param_layouts()

        self.schema_box = widgets.VBox()

        self.schema_selector.observe(self._on_schema_change, names="value")
        self.mode_selector.observe(self._update_actions, names="value")
        self.action_selector.observe(self._update_param_inputs, names="value")
        self.param_view_toggle.observe(self._update_param_inputs, names="value")
        self.tmux_run_button.on_click(self._tmux_execute)
        self.inline_run_button.on_click(self._inline_execute)

        self.ui = widgets.VBox([
            self.schema_selector,
            self.schema_box,
            self.mode_selector,
            self.mode_description,
            self.action_selector,
            self.param_view_toggle,
            self.param_box,
            self.tmux_run_button,
            self.inline_run_button,
            self.output,
        ])

        self._on_schema_change()
        self._update_actions()
        self._update_mode_description()
        self._update_param_inputs()

    # schema panels
    def _build_all_schema_layouts(self):
        """Construct UI rows for each DTO schema and cache their widget maps."""
        all_subjects = self.controller.list_subjects()
        all_tasks = self.controller.list_all_tasks()

        for schema_dto in self.schemas:
            rows, wmap = [], {}
            subject_schema = is_subject_schema(schema_dto)

            schema_fields = list(fields(schema_dto))
            if subject_schema:
                schema_fields.sort(
                    key=lambda field: (0 if field.name == 'subject' else (1 if field.name == 'task' else 2)))

            for f in schema_fields:
                name = f.name
                if subject_schema and name == 'run':
                    continue

                label = widgets.Label(value=f"{name}:", layout=widgets.Layout(width='200px'))
                if subject_schema and name == "subject":
                    w = widgets.Dropdown(options=all_subjects, layout=widgets.Layout(width='220px'))
                    w.observe(lambda change, _w=w, _schema=schema_dto: self._on_subject_changed(_schema), names="value")
                elif subject_schema and name == "task":
                    w = widgets.Dropdown(options=[], layout=widgets.Layout(width='220px'))
                elif (not subject_schema) and name == "task":
                    w = widgets.Dropdown(options=all_tasks, layout=widgets.Layout(width='220px'))
                elif (schema_dto is SubjectFilterDTO) and name.endswith('_range'):
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
                self._refresh_task_options(schema_dto, init=True)

    def _on_subject_changed(self, schema_dto: type[BaseTaskDTO]):
        """Handle subject dropdown changes by refreshing task options and params."""
        self._refresh_task_options(schema_dto, init=False)
        self._update_param_inputs()

    def _refresh_task_options(self, schema_dto: type[BaseTaskDTO], init=False):
        """Populate the task dropdown for the selected subject; preserve selection on init when possible."""
        wmap = self.schema_widgets.get(schema_dto)
        subj_w, task_w = wmap.get("subject"), wmap.get("task")
        if subj_w is None or task_w is None:
            return
        subj = subj_w.value
        tasks = self.controller.list_tasks(subj)
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

    def _build_all_param_layouts(self):
        """Build parameter input widgets for every registered action in specs.

        New behavior: group fields by defining class and render later as tabs.
        """
        for group in self.specs:
            for key, spec in self.specs[group].items():
                params_cls = spec.get("params")
                layout_key = (group, key)
                widgets_dict = {}
                group_meta = []  # list of {owner,title,field_names,rows}
                if params_cls:
                    params_obj = params_cls() if callable(params_cls) else params_cls
                    # Derive grouped field ownership
                    groups = derive_param_groups(params_obj)
                    for g in groups:
                        owner = g["owner"]
                        title = g["title"]
                        field_names = g["field_names"]
                        rows_for_owner = []
                        pair_buf = []
                        # Order fields within this group using ordered_fields(owner)
                        try:
                            ordered_owner_fields = [f.name for f in ordered_fields(owner)]
                        except Exception:
                            ordered_owner_fields = field_names
                        ordered_names = [n for n in ordered_owner_fields if n in field_names]
                        if not ordered_names:
                            ordered_names = field_names
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

                # Flatten legacy layout (concatenate) for backward compatibility in any code expecting param_layouts
                flat_rows = []
                for gm in group_meta:
                    flat_rows.extend(gm["rows"])
                self.param_layouts[layout_key] = flat_rows
                self.param_widgets[layout_key] = widgets_dict
                self.param_group_meta[layout_key] = group_meta

    def _on_schema_change(self, *_):
        """Switch schema panel when the input type toggle changes."""
        schema_dto = self.schema_selector.value
        self.schema_box.children = self.schema_layouts.get(schema_dto, [])
        self.schema_box.layout.display = None if self.schema_box.children else "none"
        if is_subject_schema(schema_dto):
            self._refresh_task_options(schema_dto, init=False)
        self._update_param_inputs()

    def _current_subject_schema_dto_for_prepare(self):
        """Return a TaskDTO-like instance for prepare() when in single-subject mode."""
        schema_dto = self.schema_selector.value
        if not is_subject_schema(schema_dto):
            return None
        wmap = self.schema_widgets.get(schema_dto, {})
        subject = wmap["subject"].value
        task, run = wmap["task"].value
        return schema_dto(subject=subject, task=task, run=run)

    def _update_actions(self, *_):
        """Refresh actions list based on selected mode and pick the first action."""
        group = self.mode_selector.value
        self.action_selector.options = list(self.specs[group].keys())
        if self.action_selector.options:
            self.action_selector.value = self.action_selector.options[0]
        self._update_param_inputs()
        self._update_mode_description()

    def _update_mode_description(self):
        group = self.mode_selector.value
        desc = (self.mode_info or {}).get(group, "")
        if desc:
            self.mode_description.value = f"<div style='color:#666;font-size:12px;margin:4px 0 8px 0'>{desc}</div>"
        else:
            self.mode_description.value = ""

    def _update_param_inputs(self, *_):
        """Rebuild the parameter panel and apply dynamic defaults via controller.prepare."""
        group = self.mode_selector.value
        key = self.action_selector.value
        if not group or not key:
            return
        layout_key = (group, key)
        group_meta = self.param_group_meta.get(layout_key, [])
        # Build UI according to toggle: tabs (default) vs show-all-with-headers
        if group_meta:
            show_all = bool(getattr(self.param_view_toggle, "value", False))
            if show_all:
                blocks = []
                for gm in group_meta:
                    header = widgets.HTML(
                        value=f"<div style='font-weight:600;margin:6px 0 4px 0'>{gm['title']}</div>"
                    )
                    blocks.append(header)
                    blocks.extend(gm["rows"])  # reuse existing widgets
                self.param_box.children = blocks
            else:
                tab = widgets.Tab()
                tab_children = []
                for gm in group_meta:
                    tab_children.append(widgets.VBox(gm["rows"]))
                tab.children = tab_children
                for idx, gm in enumerate(group_meta):
                    tab.set_title(idx, gm["title"])
                self.param_box.children = [tab]
        else:
            self.param_box.children = self.param_layouts.get(layout_key, [])
        self.param_inputs = self.param_widgets.get(layout_key, {})
        task_dto = self._current_subject_schema_dto_for_prepare()
        if task_dto is None:
            return
        updates = self.controller.prepare(task_dto, group, key) or {}
        for name, new_val in updates.items():
            w = self.param_inputs.get(name)
            if w is None:
                continue
            if isinstance(w, widgets.Dropdown):
                w.options = new_val
                if new_val:
                    w.value = new_val[0]
            elif hasattr(w, "value"):
                w.value = new_val

    def _build_active_dto(self):
        """Create the active DTO instance by reading all schema widgets."""
        schema_dto = self.schema_selector.value
        wmap = self.schema_widgets[schema_dto]
        kwargs = {}
        subject_schema = is_subject_schema(schema_dto)
        for f in fields(schema_dto):
            name = f.name
            if subject_schema and name == "run":
                continue
            if name not in wmap:
                continue
            default_val = field_default(f)
            wrap_list = (not subject_schema and isinstance(default_val, list))
            val = read_widget(wmap[name], default_val, wrap_list=wrap_list, field=f)
            if subject_schema and name == "task":
                if isinstance(val, (tuple, list)) and len(val) == 2:
                    kwargs["task"], kwargs["run"] = val[0], val[1]
                else:
                    kwargs["task"], kwargs["run"] = val, None
            else:
                kwargs[name] = val
        return schema_dto(**kwargs)

    def _collect_params(self, group: str, key: str, spec: dict):
        """Construct params DTO from widgets and defaults."""
        params_cls = spec.get("params")
        if not params_cls:
            return None

        defaults = params_cls()
        widgets_map = self.param_widgets.get((group, key), {})
        values = {}
        for f in fields(defaults):
            values[f.name] = read_widget(
                widgets_map.get(f.name), getattr(defaults, f.name), wrap_list=False, field=f
            )
        return params_cls(**values)

    def _prepare_execution(self):
        """Builds all required inputs for execution (dto, group, key, params)."""
        group = self.mode_selector.value
        key = self.action_selector.value
        spec = self.specs[group][key]
        dto = self._build_active_dto()
        params_dto = self._collect_params(group, key, spec)
        return dto, group, key, params_dto

    def _tmux_execute(self, _):
        """Schedule the selected action as a background job using JobRunner."""
        with self.output:
            clear_output(wait=True)
            print("Scheduling job...")

            dto, group, key, params_dto = self._prepare_execution()
            runner = JobRunner(self.controller, self.jobs_root)
            runner.schedule(group, key, dto, params_dto)

            self._update_param_inputs()

    def _inline_execute(self, _):
        """Execute synchronously in the notebook."""
        with self.output:
            clear_output(wait=True)
            print("Execute job...")

            dto, group, key, params_dto = self._prepare_execution()
            result = self.controller.show(dto, group, key, params_dto)

            # Unified display handling
            if isinstance(result, pd.DataFrame):
                display(result)
            elif isinstance(result, plt.Figure):
                display(result)
            elif isinstance(result, list) and all(isinstance(fig, plt.Figure) for fig in result):
                for fig in result:
                    display(fig)
            elif isinstance(result, (dict, list)):
                print(json.dumps(result, indent=2))
            elif isinstance(result, str):
                print(result)
            elif result is not None:
                print("Output:", result)

            self._update_param_inputs()

    def show(self):
        """Display the assembled UI in the current notebook output cell."""
        display(self.ui)
