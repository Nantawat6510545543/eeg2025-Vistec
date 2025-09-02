import ipywidgets as widgets
from IPython.display import display, clear_output
from dataclasses import fields, is_dataclass
import pandas as pd
import matplotlib.pyplot as plt

from ..models import TaskDTO, SubjectFilterDTO

plt.ioff()


class EEGUI:
    def __init__(self, controller):
        self.controller = controller
        self.specs = self.controller.get_specs()

        # Input type buttons
        input_options = [
            (getattr(TaskDTO, "ui_name", "Single subject"), TaskDTO),
            (getattr(SubjectFilterDTO, "ui_name", "Meta filter (group)"), SubjectFilterDTO),
        ]
        self.input_type_buttons = widgets.ToggleButtons(
            options=input_options, description="Input Type:", layout=widgets.Layout(width='auto')
        )
        self.input_type_buttons.value = input_options[0][1]

        # Basic controls
        self.mode_selector = widgets.ToggleButtons(options=list(self.specs.keys()), description="Mode:")
        self.action_selector = widgets.ToggleButtons(description="Action:")
        self.param_box = widgets.VBox()
        self.run_button = widgets.Button(description="Run", button_style="success")
        self.output = widgets.Output()

        # Subject / Task selectors
        self.subject_dropdown = widgets.Dropdown(options=self.controller.list_subjects(), description="Subject:")
        self.task_dropdown = widgets.Dropdown(description="Task:")

        # Filter panel (only shown for SubjectFilterDTO)
        self.filter_box = widgets.VBox()
        self.filter_widgets = {}

        # Params UI for actions
        self.param_layouts = {}
        self.param_widgets = {}
        self._build_all_param_layouts()

        # Wire events
        self.input_type_buttons.observe(self._on_input_type_change, names="value")
        self.subject_dropdown.observe(self._update_tasks, names="value")
        self.task_dropdown.observe(self._update_param_inputs, names="value")
        self.mode_selector.observe(self._update_actions, names="value")
        self.action_selector.observe(self._update_param_inputs, names="value")
        self.run_button.on_click(self._execute)

        # Layout
        self.ui = widgets.VBox([
            self.input_type_buttons,
            self.subject_dropdown,
            self.task_dropdown,
            self.filter_box,
            self.mode_selector,
            self.action_selector,
            self.param_box,
            self.run_button,
            self.output
        ])

        # Init
        self._update_tasks()
        self._update_actions()
        self._update_param_inputs()
        self._rebuild_filter_panel()

    # -----------------------
    # Simple UI builders
    # -----------------------
    def _make_widget(self, value):
        if isinstance(value, tuple) and len(value) == 2 and all(isinstance(x, (int, float)) for x in value):
            vmin, vmax = float(value[0]), float(value[1])
            return {"min": widgets.FloatText(value=vmin, layout=widgets.Layout(width='100px')),
                    "max": widgets.FloatText(value=vmax, layout=widgets.Layout(width='100px'))}
        if isinstance(value, bool):
            return widgets.Checkbox(value=value, layout=widgets.Layout(width='150px'))
        if isinstance(value, int):
            return widgets.IntText(value=value, layout=widgets.Layout(width='150px'))
        if isinstance(value, float):
            return widgets.FloatText(value=value, layout=widgets.Layout(width='150px'))
        if isinstance(value, list):
            return widgets.Dropdown(options=value, layout=widgets.Layout(width='150px'))
        return widgets.Text(value="" if value is None else str(value), layout=widgets.Layout(width='150px'))

    def _read_widget(self, widget, default, wrap_list=False):
        if isinstance(widget, dict):  # range pair
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


    # -----------------------
    # Filter panel (only for SubjectFilterDTO)
    # -----------------------
    def _rebuild_filter_panel(self):
        cls = self.input_type_buttons.value
        if cls is TaskDTO:
            self.subject_dropdown.layout.display = None
            self.filter_box.children = []
            self.filter_widgets = {}
            self.filter_box.layout.display = "none"
            return

        # SubjectFilterDTO UI
        self.subject_dropdown.layout.display = "none"
        self.filter_box.layout.display = None

        # Build widgets from SubjectFilterDTO defaults
        defaults = SubjectFilterDTO(task="")  # only 'task' required; defaults feed the rest
        rows, widgets_map = [], {}
        for f in fields(SubjectFilterDTO):
            if f.name in ("task", "run", "subject"):
                continue
            default_val = getattr(defaults, f.name)
            w = self._make_widget(default_val)
            widgets_map[f.name] = w
            label = widgets.Label(value=f"{f.name}:", layout=widgets.Layout(width='200px'))
            if isinstance(w, dict):
                pair = widgets.HBox([
                    widgets.Label("min", layout=widgets.Layout(width="40px")), w["min"],
                    widgets.Label("max", layout=widgets.Layout(width="40px")), w["max"],
                ])
                rows.append(widgets.HBox([label, pair]))
            else:
                rows.append(widgets.HBox([label, w]))
        self.filter_widgets = widgets_map
        self.filter_box.children = rows

    def _on_input_type_change(self, *_):
        self._rebuild_filter_panel()
        self._update_param_inputs()

    # -----------------------
    # Action params UI
    # -----------------------
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

    # -----------------------
    # Controller plumbing
    # -----------------------
    def _update_tasks(self, *_):
        subject = self.subject_dropdown.value
        task_keys = self.controller.list_tasks(subject)
        self.task_dropdown.options = [(f"{t} (Run {r})" if r else t, (t, r)) for t, r in task_keys]
        if self.task_dropdown.options:
            self.task_dropdown.value = self.task_dropdown.options[0][1]
        self._update_param_inputs()

    def _update_actions(self, *_):
        group = self.mode_selector.value
        self.action_selector.options = list(self.specs[group].keys())
        if self.action_selector.options:
            self.action_selector.value = self.action_selector.options[0]
        self._update_param_inputs()

    def _update_param_inputs(self, *_):
        group = self.mode_selector.value
        key = self.action_selector.value
        if not group or not key or not self.task_dropdown.value:
            return
        subject = self.subject_dropdown.value
        task, run = self.task_dropdown.value
        task_dto = TaskDTO(subject=subject, task=task, run=run)
        updates = self.controller.prepare(task_dto, group, key)

        layout_key = (group, key)
        self.param_box.children = self.param_layouts.get(layout_key, [])
        self.param_inputs = self.param_widgets.get(layout_key, {})

        if updates:
            for name, new_val in updates.items():
                w = self.param_inputs.get(name)
                if w is None:
                    continue
                if isinstance(w, widgets.Dropdown):
                    w.options = new_val
                    w.value = new_val[0]
                else:
                    if hasattr(w, "value"):
                        w.value = new_val

    # -----------------------
    # Run
    # -----------------------
    def _execute(self, _):
        with self.output:
            clear_output(wait=True)
            group = self.mode_selector.value
            key = self.action_selector.value
            spec = self.specs[group][key]
            params_cls = spec["params"]

            dto_cls = self.input_type_buttons.value
            t, r = self.task_dropdown.value

            if dto_cls is TaskDTO:
                built_dto = TaskDTO(subject=self.subject_dropdown.value, task=t, run=r)
            else:
                defaults = SubjectFilterDTO(task=t)
                kwargs = {"task": t}
                for f in fields(SubjectFilterDTO):
                    if f.name in ("task", "run", "subject"):
                        continue
                    kwargs[f.name] = self._read_widget(
                        self.filter_widgets[f.name],
                        getattr(defaults, f.name),
                        wrap_list=True,   # <— only here
                    )
                built_dto = SubjectFilterDTO(**kwargs)

            # Build action params DTO (if any)
            if params_cls:
                params_defaults = params_cls()
                params_values = {f.name: self._read_widget(self.param_widgets[(group, key)][f.name],
                                                            getattr(params_defaults, f.name))
                                    for f in fields(params_defaults)}
                params_dto = params_cls(**params_values)
            else:
                params_dto = None

            result = None
            if isinstance(built_dto, TaskDTO):
                print(built_dto.subject)
                result = self.controller.show(built_dto, group, key, params_dto)
            else:
                print("Filter DTO selected:", built_dto)
                print("→ Wire this to controller's cohort path next.")

            if isinstance(result, pd.DataFrame):
                display(result)
            elif isinstance(result, list) and all(isinstance(fig, plt.Figure) for fig in result):
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
