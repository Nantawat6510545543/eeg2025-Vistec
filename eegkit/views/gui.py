import ipywidgets as widgets
from IPython.display import display, clear_output
from dataclasses import fields
import pandas as pd
from ..models import TaskDTO

class EEGUI:
    def __init__(self, controller):
        self.controller = controller
        self.specs = self.controller.get_specs()

        self.mode_selector = widgets.ToggleButtons(
            options=list(self.specs.keys()), description="Mode:"
        )
        self.action_selector = widgets.ToggleButtons(description="Action:")
        self.param_box = widgets.VBox()
        self.plot_button = widgets.Button(description="Run", button_style="success")
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
        self.plot_button.on_click(self._execute)

        self.ui = widgets.VBox([
            self.subject_dropdown,
            self.task_dropdown,
            self.mode_selector,
            self.action_selector,
            self.param_box,
            self.plot_button,
            self.output
        ])

        self._update_tasks()
        self._update_actions()  # Initialize first

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
        action = self.action_selector.value
        spec = self.specs[group][action]

        self.param_inputs.clear()
        widgets_list = []
        params_cls = spec["params"]

        if params_cls:
            for f in fields(params_cls):
                w = self._create_widget(f)
                label = widgets.Label(value=f"{f.name}:", layout=widgets.Layout(width='120px'))
                hbox = widgets.HBox([label, w])
                widgets_list.append(hbox)
                self.param_inputs[f.name] = w

        rows = [widgets.HBox(widgets_list[i:i+2]) for i in range(0, len(widgets_list), 2)]
        self.param_box.children = rows

    def _create_widget(self, f):
        from dataclasses import MISSING
        typ = f.type
        default = f.default if f.default != MISSING else None

        if typ == float:
            return widgets.FloatText(value=default or 0.0, layout=widgets.Layout(width='150px'))
        elif typ == int:
            return widgets.IntText(value=default or 0, layout=widgets.Layout(width='150px'))
        elif typ == bool:
            return widgets.Checkbox(value=default or False, layout=widgets.Layout(width='150px'))
        else:
            options = default if isinstance(default, list) else [default] if default else []
            return widgets.Dropdown(options=options, layout=widgets.Layout(width='150px'))

    def _build_dto(self, cls):
        values = {}
        for f in fields(cls):
            widget = self.param_inputs.get(f.name)
            if widget is None:
                continue
            val = widget.value
            try:
                if f.type == float:
                    val = float(val)
                elif f.type == int:
                    val = int(val)
                elif f.type == bool:
                    val = bool(val)
                elif "List" in str(f.type):
                    val = eval(val)
            except:
                pass
            values[f.name] = val
        return cls(**values)

    def _execute(self, _):
        with self.output:
            clear_output(wait=True)

            # 1. Get spec group and key (e.g., "plot_specs" / "time")
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

            if isinstance(result, pd.DataFrame):
                display(result)
            elif isinstance(result, (dict, list)):
                import json
                print(json.dumps(result, indent=2))
            elif isinstance(result, str):
                print(result)
            elif result is not None:
                print("Output:", result)

            return result


    def show(self):
        display(self.ui)
