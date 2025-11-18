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
from pathlib import Path

import ipywidgets as widgets
from IPython.display import display, clear_output

from .job_runner import JobRunner
from .ui.actions_bar import ActionsBar
from .ui.execution_panel import ExecutionPanel
from .ui.param_panel import ParamPanel
from .ui.progress_area import ProgressArea
from .ui.schema_panel import SchemaPanel
from .ui.widgets import (
    is_subject_schema,
)
from .ui_support.cache import UIParamCache
from ..models.dtos import TaskDTO, SubjectFilterDTO


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

        self.actions = ActionsBar(self.specs, self.mode_info)
        self.param_box = widgets.VBox()
        self.execution = ExecutionPanel()

        self.jobs_root = Path("jobs")
        self.jobs_root.mkdir(exist_ok=True, parents=True)

        self.schema_panel = SchemaPanel(
            self.controller,
            self.schemas,
            on_subject_change=self._on_schema_subject_changed,
        )

        self.ui_param_cache = UIParamCache()
        self.param_panel = ParamPanel(self.controller, self.ui_param_cache)
        self._build_all_param_layouts()

        self.schema_box = widgets.VBox()

        self.schema_selector.observe(self._on_schema_change, names="value")
        self.actions.mode_selector.observe(self._update_actions, names="value")
        self.actions.action_selector.observe(self._update_param_inputs, names="value")
        self.actions.param_view_toggle.observe(self._update_param_inputs, names="value")
        self.execution.tmux_button.on_click(self._tmux_execute)
        self.execution.inline_button.on_click(self._inline_execute)

        self.ui = widgets.VBox([
            self.schema_selector,
            self.schema_box,
            self.actions.container,
            self.param_box,
            self.execution.container,
        ])

        self._on_schema_change()
        self._update_actions()
        self.actions.update_mode_description()
        self._update_param_inputs()

    def _on_schema_subject_changed(self):
        """Callback from SchemaPanel when subject changes."""
        self._update_param_inputs()

    def _build_all_param_layouts(self):
        """Build parameter input widgets for every registered action in specs."""
        total = sum(len(actions) for actions in self.specs.values())
        pa = None
        built = 0
        if total > 0:
            pa = ProgressArea()
            pa.begin(total, title="Preparing parameter layouts…")

        for group in self.specs:
            for key, spec in self.specs[group].items():
                params_cls = spec.get("params")
                # Build via ParamPanel
                self.param_panel.ensure_layout(group, key, params_cls)

                if pa is not None:
                    built += 1
                    pa.update(built, total, group=group, key=key)

        if pa is not None:
            per_group = ", ".join(f"{g}: {len(actions)}" for g, actions in self.specs.items())
            summary = f"Ready. Built {built} layout(s)." + (f" Groups → {per_group}" if per_group else "")
            pa.finish(summary)

    def _apply_overlay(self, widgets_map, overlay):
        for name, val in overlay.items():
            w = widgets_map.get(name)
            if w is None:
                continue
            if isinstance(w, dict):
                if isinstance(val, (tuple, list)) and len(val) == 2:
                    vmin, vmax = val
                    if 'min' in w and hasattr(w['min'], 'value'):
                        w['min'].value = '' if vmin is None else str(vmin)
                    if 'max' in w and hasattr(w['max'], 'value'):
                        w['max'].value = '' if vmax is None else str(vmax)
            elif isinstance(w, widgets.Dropdown):
                opts = w.options or []
                values = [v if not isinstance(v, tuple) else v[1] for v in opts]
                if val in values:
                    w.value = val
            elif hasattr(w, 'value'):
                w.value = val

    def _on_schema_change(self, *_):
        """Switch schema panel when the input type toggle changes."""
        schema_dto = self.schema_selector.value
        self.schema_box.children = self.schema_panel.schema_layouts.get(schema_dto, [])
        self.schema_box.layout.display = None if self.schema_box.children else "none"
        if is_subject_schema(schema_dto):
            self.schema_panel.refresh_task_options(schema_dto, init=False)
        self._update_param_inputs()

    def _current_subject_schema_dto_for_prepare(self):
        """Return a TaskDTO-like instance for prepare() when in single-subject mode."""
        schema_dto = self.schema_selector.value
        if not is_subject_schema(schema_dto):
            return None
        return self.schema_panel.build_dto(schema_dto)

    def _update_actions(self, *_):
        """Refresh actions list based on selected mode and update UI accordingly."""
        self.actions.update_actions()
        self._update_param_inputs()
        self.actions.update_mode_description()

    def _update_param_inputs(self, *_):
        """Rebuild the parameter panel and apply dynamic defaults via controller.prepare."""
        group = self.actions.mode_selector.value
        key = self.actions.action_selector.value
        if not group or not key:
            return
        layout_key = (group, key)
        group_meta = self.param_panel.param_group_meta.get(layout_key, [])
        if group_meta:
            show_all = bool(getattr(self.actions.param_view_toggle, "value", False))
            if show_all:
                blocks = []
                for gm in group_meta:
                    header = widgets.HTML(
                        value=f"<div style='font-weight:600;margin:6px 0 4px 0'>{gm['title']}</div>"
                    )
                    blocks.append(header)
                    blocks.extend(gm["rows"])
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
            self.param_box.children = self.param_panel.param_layouts.get(layout_key, [])
        self.param_inputs = self.param_panel.param_widgets.get(layout_key, {})
        task_dto = self._current_subject_schema_dto_for_prepare()
        if task_dto is None:
            return
        spec = self.specs[group][key]
        params_dto = self.param_panel.collect_params(group, key, spec, self.param_panel.param_widgets)
        updates = self.controller.prepare(task_dto, group, key, params_dto) or {}
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
        group_meta = self.param_panel.param_group_meta.get(layout_key, [])
        overlay = self.ui_param_cache.get_overlay(group_meta)
        # Apply cached min/max overlays
        if overlay:
            self._apply_overlay(self.param_inputs, overlay)

    def _prepare_execution(self):
        """Builds all required inputs for execution (dto, group, key, params)."""
        group = self.actions.mode_selector.value
        key = self.actions.action_selector.value
        spec = self.specs[group][key]
        dto = self.schema_panel.build_dto(self.schema_selector.value)
        params_dto = self.param_panel.collect_params(group, key, spec, self.param_panel.param_widgets)
        return dto, group, key, params_dto

    def _tmux_execute(self, _):
        """Schedule the selected action as a background job using JobRunner."""
        with self.execution.output:
            clear_output(wait=True)
            print("Scheduling job...")

            dto, group, key, params_dto = self._prepare_execution()
            runner = JobRunner(self.controller, self.jobs_root)
            runner.schedule(group, key, dto, params_dto)

            self._update_param_inputs()

    def _inline_execute(self, _):
        """Execute synchronously in the notebook."""
        with self.execution.output:
            clear_output(wait=True)
            print("Execute job...")

        dto, group, key, params_dto = self._prepare_execution()
        result = self.controller.show(dto, group, key, params_dto)
        self.execution.render_result(result)
        self._update_param_inputs()

    def show(self):
        """Display the assembled UI in the current notebook output cell."""
        display(self.ui)
