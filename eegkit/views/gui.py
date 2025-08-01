import ipywidgets as widgets
from IPython.display import display, clear_output
from dataclasses import fields
import pandas as pd
from ..models import TaskDTO
import matplotlib.pyplot as plt
from dataclasses import MISSING

class EEGUI:
    def __init__(self, controller):
        self.controller = controller
        self.specs = self.controller.get_specs()

        self.mode_selector = widgets.ToggleButtons(
            options=list(self.specs.keys()), description="Mode:"
        )
        self.action_selector = widgets.ToggleButtons(description="Action:")
        self.param_box = widgets.VBox()
        self.run_button = widgets.Button(description="Run", button_style="success")
        self.output = widgets.Output()

        self.subject_dropdown = widgets.Dropdown(
            options=self.controller.list_subjects(), description="Subject:"
        )

        self.task_dropdown = widgets.Dropdown(
            description="Task:"
        )

        self.param_inputs = {}  # field_name → widget

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
        spec = self.specs[group][key]
        params_obj = spec["params"]

        self.param_inputs.clear()
        widgets_list = []

        if params_obj:
            for f in fields(params_obj):
                widget = self._create_widget(f, params_obj)
                label = widgets.Label(value=f"{f.name}:", layout=widgets.Layout(width='200px'))
                hbox = widgets.HBox([label, widget])
                widgets_list.append(hbox)
                self.param_inputs[f.name] = widget

        rows = [widgets.HBox(widgets_list[i:i+2]) for i in range(0, len(widgets_list), 2)]
        self.param_box.children = rows

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

    def _build_dto(self, params_obj):
        cls = type(params_obj)
        values = {
            f.name: self._parse_widget_value(self.param_inputs[f.name], f.type)
            for f in fields(params_obj)
        }
        return cls(**values)

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

            # 1. Get spec group and key
            group = self.mode_selector.value
            key = self.action_selector.value
            spec = self.specs[group][key]
            params_cls = spec["params"]

            # 2. Build TaskDTO from dropdowns
            subject = self.subject_dropdown.value
            task, run = self.task_dropdown.value
            task_dto = TaskDTO(subject=subject, task=task, run=run)

            # 3. Build DTO from inputs or pass None
            if params_cls is None:
                params_dto = None
            else:
                params_dto = self._build_dto(params_cls)

            # 4. Call controller with DTOs
            result = self.controller.show(task_dto, group, key, params_dto)
            
            plt.ioff
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

            return


    def show(self):
        display(self.ui)
