import ipywidgets as widgets
from IPython.display import display, clear_output
from dataclasses import fields, MISSING
import matplotlib.pyplot as plt
from .job_runner import JobRunner
from pathlib import Path

from ..models import BaseTaskDTO, TaskDTO, SubjectFilterDTO

plt.ioff()


class EEGUI:
    def __init__(self, controller):
        self.controller = controller
        self.specs = self.controller.get_specs()

        self.schemas = [TaskDTO, SubjectFilterDTO]
        self.schema_selector = widgets.ToggleButtons(
            options=[(getattr(c, "ui_name", c.__name__), c) for c in self.schemas],
            description="Input Type:", layout=widgets.Layout(width="auto")
        )
        self.schema_selector.value = self.schemas[0]

        self.mode_selector = widgets.ToggleButtons(options=list(self.specs.keys()), description="Mode:")
        self.action_selector = widgets.ToggleButtons(description="Action:")
        self.param_box = widgets.VBox()
        self.run_button = widgets.Button(description="Run", button_style="success")
        self.output = widgets.Output()
        
        self.jobs_root = Path("jobs")
        self.jobs_root.mkdir(exist_ok=True, parents=True)

        self.schema_layouts = {}
        self.schema_widgets = {}
        self._build_all_schema_layouts()

        self.param_layouts = {}
        self.param_widgets = {}
        self._build_all_param_layouts()

        self.schema_box = widgets.VBox()

        self.schema_selector.observe(self._on_schema_change, names="value")
        self.mode_selector.observe(self._update_actions, names="value")
        self.action_selector.observe(self._update_param_inputs, names="value")
        self.run_button.on_click(self._execute)

        self.ui = widgets.VBox([
            self.schema_selector,
            self.schema_box,
            self.mode_selector,
            self.action_selector,
            self.param_box,
            self.run_button,
            self.output,
        ])

        self._on_schema_change()
        self._update_actions()
        self._update_param_inputs()

    def _is_subject_schema(self, schema_dto: BaseTaskDTO):
        return any(f.name == "subject" for f in fields(schema_dto))

    def _field_default(self, f):
        if f.default is not MISSING:
            return f.default
        if getattr(f, "default_factory", MISSING) is not MISSING:
            try:
                return f.default_factory()
            except Exception:
                return None
        return None

    def _make_widget(self, value):
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

    def _make_range_widget(self, default_value):
        """Create a range widget supporting open-ended bounds.
        default_value expected: (lower, upper) where each may be None.
        """
        if isinstance(default_value, (tuple, list)) and len(default_value) == 2:
            lower, upper = default_value
        else:
            lower, upper = (None, None)
        return {
            "min": widgets.Text(value='' if lower is None else str(lower), placeholder='min', layout=widgets.Layout(width='100px')),
            "max": widgets.Text(value='' if upper is None else str(upper), placeholder='max', layout=widgets.Layout(width='100px')),
        }

    def _read_widget(self, widget, default, wrap_list=False):
        if isinstance(widget, dict):
            vmin = widget["min"].value
            vmax = widget["max"].value
            # Interpret blanks as open bounds
            lower = None if vmin in (None, "") else float(vmin)
            upper = None if vmax in (None, "") else float(vmax)
            if isinstance(default, (tuple, list)) and len(default) == 2:
                # If both open and default is open, keep open
                if lower is None and upper is None:
                    return (None, None)
            return (lower, upper)
        val = getattr(widget, "value", None)
        if wrap_list and isinstance(default, list):
            return [val]
        if isinstance(default, bool):
            return bool(val)
        if isinstance(default, int):
            try:
                return int(val)
            except Exception:
                return default
        if isinstance(default, float):
            try:
                return float(val)
            except Exception:
                return default
        return val
    
    # schema panels
    def _build_all_schema_layouts(self):
        all_subjects = self.controller.list_subjects()
        all_tasks = self.controller.list_all_tasks()

        for schema_dto in self.schemas:
            rows, wmap = [], {}
            subject_schema = self._is_subject_schema(schema_dto)

            for f in fields(schema_dto):
                name = f.name
                label = widgets.Label(value=f"{name}:", layout=widgets.Layout(width='200px'))

                if subject_schema and name == "subject":
                    w = widgets.Dropdown(options=all_subjects, layout=widgets.Layout(width='220px'))
                    w.observe(lambda change, _w=w: self._on_subject_changed(schema_dto), names="value")
                elif subject_schema and name == "task":
                    w = widgets.Dropdown(options=[], layout=widgets.Layout(width='220px'))
                elif subject_schema and name == "run":
                    continue
                elif (not subject_schema) and name == "task":
                    w = widgets.Dropdown(options=all_tasks, layout=widgets.Layout(width='220px'))
                elif (schema_dto is SubjectFilterDTO) and name.endswith('_range'):
                    default_val = self._field_default(f)
                    w = self._make_range_widget(default_val)
                else:
                    default_val = self._field_default(f)
                    w = self._make_widget(default_val)

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

    def _on_subject_changed(self, schema_dto: BaseTaskDTO):
        self._refresh_task_options(schema_dto, init=False)
        self._update_param_inputs()

    def _refresh_task_options(self, schema_dto: BaseTaskDTO, init=False):
        wmap = self.schema_widgets.get(schema_dto)
        subj_w, task_w = wmap.get("subject"), wmap.get("task")
        subj = subj_w.value
        tasks = self.controller.list_tasks(subj)
        opts = []
        for t, r in tasks:
            label = f"{t} (Run {r})" if r else f"{t}"
            opts.append((label, (t, r)))
        task_w.options = opts
        if opts and (init or task_w.value not in [v for _, v in opts]):
            task_w.value = opts[0][1]

    # action params
    def _build_all_param_layouts(self):
        for group in self.specs:
            for key, spec in self.specs[group].items():
                params_cls = spec["params"]
                layout_key = (group, key)
                widgets_dict = {}
                rows, pair_buf = [], []
                if params_cls:
                    params_obj = params_cls() if callable(params_cls) else params_cls
                    for f in fields(params_obj):
                        val = getattr(params_obj, f.name)
                        w = self._make_widget(val)
                        widgets_dict[f.name] = w
                        row = widgets.HBox([
                            widgets.Label(value=f"{f.name}:", layout=widgets.Layout(width='200px')),
                            w
                        ])
                        if isinstance(val, str):
                            if pair_buf:
                                rows.append(widgets.HBox(pair_buf))
                                pair_buf = []
                            rows.append(row)
                        else:
                            pair_buf.append(row)
                            if len(pair_buf) == 2:
                                rows.append(widgets.HBox(pair_buf))
                                pair_buf = []

                if pair_buf:
                    rows.append(widgets.HBox(pair_buf))
                self.param_layouts[layout_key] = rows
                self.param_widgets[layout_key] = widgets_dict

    # switching & updates
    def _on_schema_change(self, *_):
        schema_dto = self.schema_selector.value
        self.schema_box.children = self.schema_layouts.get(schema_dto, [])
        self.schema_box.layout.display = None if self.schema_box.children else "none"
        if self._is_subject_schema(schema_dto):
            self._refresh_task_options(schema_dto, init=False)
        self._update_param_inputs()

    def _current_subject_schema_dto_for_prepare(self):
        schema_dto = self.schema_selector.value
        if not self._is_subject_schema(schema_dto):
            return None
        wmap = self.schema_widgets.get(schema_dto, {})
        subject = wmap["subject"].value
        task, run = wmap["task"].value  # combined selection
        return schema_dto(subject=subject, task=task, run=run)

    def _update_actions(self, *_):
        group = self.mode_selector.value
        self.action_selector.options = list(self.specs[group].keys())
        if self.action_selector.options:
            self.action_selector.value = self.action_selector.options[0]
        self._update_param_inputs()

    def _update_param_inputs(self, *_):
        group = self.mode_selector.value
        key = self.action_selector.value
        if not group or not key:
            return
        layout_key = (group, key)
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
        schema_dto = self.schema_selector.value
        wmap = self.schema_widgets[schema_dto]
        kwargs = {}
        subject_schema = self._is_subject_schema(schema_dto)
        for f in fields(schema_dto):
            name = f.name
            if subject_schema and name == "run":
                continue
            if name not in wmap:
                continue
            default_val = self._field_default(f)
            wrap_list = (not subject_schema and isinstance(default_val, list))
            val = self._read_widget(wmap[name], default_val, wrap_list=wrap_list)
            if subject_schema and name == "task":
                if isinstance(val, (tuple, list)) and len(val) == 2:
                    kwargs["task"], kwargs["run"] = val[0], val[1]
                else:
                    kwargs["task"], kwargs["run"] = val, None
            else:
                kwargs[name] = val
        return schema_dto(**kwargs)

    def _execute(self, _):
        with self.output:
            clear_output(wait=True)
            print("Scheduling job...")

            group = self.mode_selector.value
            key = self.action_selector.value
            spec = self.specs[group][key]
            params_cls = spec["params"]

            built_dto = self._build_active_dto()
            params_dto = None
            if params_cls:
                defaults = params_cls()
                widgets_map = self.param_widgets.get((group, key), {})
                params_values = {
                    f.name: self._read_widget(widgets_map[f.name], getattr(defaults, f.name), wrap_list=False)
                    for f in fields(defaults)
                }
                params_dto = params_cls(**params_values)

            runner = JobRunner(self.controller, self.jobs_root)
            job_dir = runner.schedule(group, key, built_dto, params_dto)

            print(f"Queued: {group}/{key} for subject={getattr(built_dto, 'subject', None)}, task={getattr(built_dto, 'task', None)}")
            print(f"Job directory: {job_dir}")

            self._update_param_inputs()

    def show(self):
        display(self.ui)
