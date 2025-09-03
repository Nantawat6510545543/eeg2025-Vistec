import ipywidgets as widgets
from IPython.display import display, clear_output
from dataclasses import fields, MISSING
import pandas as pd
import matplotlib.pyplot as plt

from ..models import TaskDTO, SubjectFilterDTO

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

    def _is_subject_schema(self, cls):
        return any(f.name == "subject" for f in fields(cls))

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
        return widgets.Text(value="" if value is None else str(value), layout=widgets.Layout(width='200px'))

    def _read_widget(self, widget, default, wrap_list=False):
        if isinstance(widget, dict):
            try:
                return (float(widget["min"].value), float(widget["max"].value))
            except Exception:
                return default
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

        def subject_task_pairs(subject):
            return self.controller.list_tasks(subject)  # list[(task, run)]

        all_tasks = []
        for s in all_subjects:
            for t, _r in self.controller.list_tasks(s):
                if t not in all_tasks:
                    all_tasks.append(t)

        for cls in self.schemas:
            rows, wmap = [], {}
            subject_schema = self._is_subject_schema(cls)

            for f in fields(cls):
                name = f.name
                label = widgets.Label(value=f"{name}:", layout=widgets.Layout(width='200px'))

                if subject_schema and name == "subject":
                    w = widgets.Dropdown(options=all_subjects, layout=widgets.Layout(width='220px'))
                    w.observe(lambda change, _w=w: self._on_subject_changed(cls), names="value")
                elif subject_schema and name == "task":
                    w = widgets.Dropdown(options=[], layout=widgets.Layout(width='220px'))
                elif subject_schema and name == "run":
                    continue
                elif (not subject_schema) and name == "task":
                    w = widgets.Dropdown(options=all_tasks, layout=widgets.Layout(width='220px'))
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

            self.schema_layouts[cls] = rows
            self.schema_widgets[cls] = wmap

            if subject_schema:
                self._refresh_task_options(cls, subject_task_pairs, init=True)

    def _on_subject_changed(self, cls):
        def subject_task_pairs(subject):
            return self.controller.list_tasks(subject)
        self._refresh_task_options(cls, subject_task_pairs, init=False)
        if hasattr(self, "mode_selector"):
            self._update_param_inputs()

    def _refresh_task_options(self, cls, pair_fn, init=False):
        wmap = self.schema_widgets.get(cls)
        if not wmap:
            return
        subj_w, task_w = wmap.get("subject"), wmap.get("task")
        if not (subj_w and task_w):
            return
        subj = subj_w.value or (self.controller.list_subjects()[0] if self.controller.list_subjects() else None)
        pairs = pair_fn(subj) if subj is not None else []
        opts = []
        for t, r in pairs:
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
                widgets_list, widgets_dict = [], {}
                if params_cls:
                    params_obj = params_cls() if callable(params_cls) else params_cls
                    for f in fields(params_obj):
                        w = self._make_widget(getattr(params_obj, f.name))
                        widgets_list.append(widgets.HBox([
                            widgets.Label(value=f"{f.name}:", layout=widgets.Layout(width='200px')),
                            w
                        ]))
                        widgets_dict[f.name] = w
                rows = [widgets.HBox(widgets_list[i:i + 2]) for i in range(0, len(widgets_list), 2)]
                self.param_layouts[layout_key] = rows
                self.param_widgets[layout_key] = widgets_dict

    # switching & updates
    def _on_schema_change(self, *_):
        cls = self.schema_selector.value
        self.schema_box.children = self.schema_layouts.get(cls, [])
        self.schema_box.layout.display = None if self.schema_box.children else "none"
        if self._is_subject_schema(cls):
            self._refresh_task_options(cls, lambda s: self.controller.list_tasks(s), init=False)
        self._update_param_inputs()

    def _current_subject_schema_dto_for_prepare(self):
        cls = self.schema_selector.value
        if not self._is_subject_schema(cls):
            return None
        wmap = self.schema_widgets.get(cls, {})
        try:
            subject = wmap["subject"].value
            task, run = wmap["task"].value  # combined selection
            return cls(subject=subject, task=task, run=run)
        except Exception:
            return None

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
        cls = self.schema_selector.value
        wmap = self.schema_widgets[cls]
        kwargs = {}
        subject_schema = self._is_subject_schema(cls)
        for f in fields(cls):
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
        return cls(**kwargs)

    def _execute(self, _):
        with self.output:
            clear_output(wait=True)
            print("Excuting")
            group = self.mode_selector.value
            key = self.action_selector.value
            spec = self.specs[group][key]
            params_cls = spec["params"]

            built_dto = self._build_active_dto()

            params_defaults = params_cls()
            widgets_map = self.param_widgets[(group, key)]
            params_values = {
                f.name: self._read_widget(widgets_map[f.name], getattr(params_defaults, f.name), wrap_list=False)
                for f in fields(params_defaults)
            }
            params_dto = params_cls(**params_values)

            result = self.controller.show(built_dto, group, key, params_dto)

            if isinstance(result, pd.DataFrame):
                display(result)
            elif isinstance(result, list) and all(hasattr(fig, "savefig") for fig in result):
                for fig in result:
                    display(fig)
            elif isinstance(result, (dict, list)):
                import json
                print(json.dumps(result, indent=2))
            elif isinstance(result, str):
                print(result)
            elif result is not None:
                print("Output:", result)

            self._update_param_inputs()

    def show(self):
        display(self.ui)
