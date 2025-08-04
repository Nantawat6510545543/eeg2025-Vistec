import ipywidgets as widgets
from IPython.display import display, clear_output
from dataclasses import fields
import pandas as pd
from ..models import TaskDTO
import matplotlib.pyplot as plt
plt.ioff()

class EEGUI:
    def __init__(self, controller):
        self.controller = controller
        self.specs = self.controller.get_specs()

        self.mode_selector = widgets.ToggleButtons(options=list(self.specs.keys()), description="Mode:")
        self.action_selector = widgets.ToggleButtons(description="Action:")
        self.param_box = widgets.VBox()
        self.run_button = widgets.Button(description="Run", button_style="success")
        self.output = widgets.Output()

        self.subject_dropdown = widgets.Dropdown(options=self.controller.list_subjects(), description="Subject:")
        self.task_dropdown = widgets.Dropdown(description="Task:")

        self.param_layouts = {}
        self.param_widgets = {}
        self._build_all_param_layouts()

        self.mode_selector.observe(self._update_actions, names="value")
        self.action_selector.observe(self._update_param_inputs, names="value")
        self.run_button.on_click(self._execute)

        self.ui = widgets.VBox([
            self.subject_dropdown,
            self.task_dropdown,
            self.mode_selector,
            self.action_selector,
            self.param_box,
            self.run_button,
            self.output
        ])

        self._update_tasks()
        self._update_actions()

    def _build_all_param_layouts(self):
        for group in self.specs:
            for key in self.specs[group]:
                spec = self.specs[group][key]
                params_cls = spec["params"]
                layout_key = (group, key)
                widgets_list = []
                widgets_dict = {}
                if params_cls:
                    params_obj = params_cls() if callable(params_cls) else params_cls
                    for f in fields(params_obj):
                        widget = self._create_widget(f, params_obj)
                        label = widgets.Label(value=f"{f.name}:", layout=widgets.Layout(width='200px'))
                        hbox = widgets.HBox([label, widget])
                        widgets_list.append(hbox)
                        widgets_dict[f.name] = widget
                rows = [widgets.HBox(widgets_list[i:i+2]) for i in range(0, len(widgets_list), 2)]
                self.param_layouts[layout_key] = rows
                self.param_widgets[layout_key] = widgets_dict

    def _update_tasks(self, *args):
        subject = self.subject_dropdown.value
        task_keys = self.controller.list_tasks(subject)
        options = [(f"{task} (Run {run})" if run else task, (task, run)) for task, run in task_keys]
        self.task_dropdown.options = options
        if options:
            self.task_dropdown.value = options[0][1]

    def _update_actions(self, *args):
        group = self.mode_selector.value
        self.action_selector.options = list(self.specs[group].keys())
        if self.action_selector.options:
            self.action_selector.value = self.action_selector.options[0]

    def _update_param_inputs(self, *args):
        group = self.mode_selector.value
        key = self.action_selector.value
        layout_key = (group, key)

        subject = self.subject_dropdown.value
        task, run = self.task_dropdown.value
        task_dto = TaskDTO(subject=subject, task=task, run=run)
        self.controller.prepare(task_dto, group, key)

        self.param_box.children = self.param_layouts.get(layout_key, [])
        self.param_inputs = self.param_widgets.get(layout_key, {})

    def _create_widget(self, f, obj):
        value = getattr(obj, f.name)
        typ = f.type

        if typ == float:
            return widgets.FloatText(value=value or 0.0, layout=widgets.Layout(width='150px'))
        elif typ == int:
            return widgets.IntText(value=value or 0, layout=widgets.Layout(width='150px'))
        elif typ == bool:
            return widgets.Checkbox(value=value or False, layout=widgets.Layout(width='150px'))
        elif isinstance(value, list):
            return widgets.Dropdown(options=value, layout=widgets.Layout(width='150px'))
        else:
            return widgets.Text(value=str(value) if value is not None else '', layout=widgets.Layout(width='150px'))

    def _build_dto(self, params_cls):
        values = {
            f.name: self._parse_widget_value(self.param_inputs[f.name], f.type)
            for f in fields(params_cls())
        }
        return params_cls(**values)

    def _parse_widget_value(self, widget, typ):
        val = widget.value
        try:
            if typ == float:
                return float(val)
            elif typ == int:
                return int(val)
            elif typ == bool:
                return bool(val)
            elif isinstance(widget, widgets.Dropdown):
                return val
            return val
        except Exception:
            return val

    def _execute(self, _):
        with self.output:
            clear_output(wait=True)

            try:
                group = self.mode_selector.value
                key = self.action_selector.value
                spec = self.specs[group][key]
                params_cls = spec["params"]

                subject = self.subject_dropdown.value
                task, run = self.task_dropdown.value
                task_dto = TaskDTO(subject=subject, task=task, run=run)

                if params_cls is None:
                    params_dto = None
                else:
                    try:
                        params_dto = self._build_dto(params_cls)
                    except Exception as param_err:
                        print(f"[Error] Invalid parameter values: {param_err}")
                        return

                try:
                    result = self.controller.show(task_dto, group, key, params_dto)
                except Exception as func_err:
                    print(f"[Error] Failed to execute function: {func_err}")
                    return

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

            except Exception as e:
                print("[Unexpected Error]", e)

    def show(self):
        display(self.ui)
