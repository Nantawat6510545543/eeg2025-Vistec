import ipywidgets as widgets
from IPython.display import display, clear_output
import json
from .controller import EEGController

def parse_time_input(text_value):
    text_value = text_value.strip()
    return None if text_value == "" or text_value.lower() == "none" else float(text_value)

def extract_params(spec, kwargs):
    valid_keys = set(spec["params"].keys())
    return {k: kwargs[k] for k in kwargs if k in valid_keys}

class EEGUI:
    def __init__(self, controller: 'EEGController'):
        self.controller = controller
        self._init_widgets()
        self._build_ui()
        self._connect_events()
        self._initialize_state()

    def _init_widgets(self):
        subjects = sorted(self.controller.list_subjects())

        self.mode_toggle = widgets.ToggleButtons(
            options=['Plot', 'Table'], description='Mode:', layout=widgets.Layout(width='300px')
        )
        self.subject_dropdown = widgets.Dropdown(
            options=subjects, description='Subject:', layout=widgets.Layout(width='250px')
        )
        self.task_dropdown = widgets.Dropdown(description='Task:', layout=widgets.Layout(width='250px'))

        self.plot_specs = self.controller.get_plot_specs()
        self.default_params = self.controller.get_default_params()

        self.plot_type = widgets.Dropdown(
            options=list(self.plot_specs.keys()),
            description='Plot:', layout=widgets.Layout(width='300px')
        )

        self.param_inputs = {}  # name → widget
        self.param_box = widgets.VBox([])

        self.plot_button = widgets.Button(description='Plot', button_style='success')

        self.table_type = widgets.Dropdown(
            options=['events', 'channels', 'electrodes', 'epochs'],
            description='Table:', layout=widgets.Layout(width='250px')
        )
        self.rows_int = widgets.IntText(
            value=10, description='Rows:', layout=widgets.Layout(width='200px')
        )
        self.info_button = widgets.Button(description='Show Info', button_style='info')
        self.output = widgets.Output()

        self.table_param_box = widgets.VBox([])

        self.table_controls = widgets.VBox([
            widgets.HBox([self.table_type, self.rows_int, self.info_button]),
            self.table_param_box
        ])

    def _build_ui(self):
        self.ui = widgets.VBox([
            self.mode_toggle,
            self.subject_dropdown,
            self.task_dropdown,
            self.plot_type,
            self.param_box,
            self.plot_button,
            self.table_controls,
            self.output
        ])

    def _connect_events(self):
        self.mode_toggle.observe(self.update_mode_ui, names='value')
        self.subject_dropdown.observe(self.update_tasks, names='value')
        self.plot_type.observe(self.update_param_inputs, names='value')
        self.plot_button.on_click(self.do_plot)
        self.info_button.on_click(self.do_show_info)

    def _initialize_state(self):
        if self.subject_dropdown.options:
            self.subject_dropdown.value = self.subject_dropdown.options[0]
            self.update_tasks()
        self.update_param_inputs()
        self.update_mode_ui()

    def update_mode_ui(self, *args):
        is_plot = self.mode_toggle.value == 'Plot'
        self.plot_type.layout.display = 'block' if is_plot else 'none'
        self.param_box.layout.display = 'block' if is_plot else 'none'
        self.plot_button.layout.display = 'inline-block' if is_plot else 'none'
        self.table_type.layout.display = 'block' if not is_plot else 'none'
        self.info_button.layout.display = 'inline-block' if not is_plot else 'none'
        self.rows_int.layout.display = 'block' if not is_plot else 'none'
        self.table_param_box.layout.display = 'block' if not is_plot else 'none'

        if not is_plot:
            self.update_table_params()

    def update_tasks(self, *args):
        subject = self.subject_dropdown.value
        task_keys = sorted(self.controller.list_tasks(subject))
        formatted = [(f"{t} (run {r})" if r else t, (t, r)) for t, r in task_keys]
        self.task_dropdown.options = formatted
        if formatted:
            self.task_dropdown.value = formatted[0][1]

    def _create_widget(self, param_type, default):
        if param_type == "float":
            return widgets.FloatText(value=default, layout=widgets.Layout(width='150px'))
        elif param_type == "int":
            return widgets.IntText(value=default, layout=widgets.Layout(width='150px'))
        elif param_type == "str":
            return widgets.Text(value=default, layout=widgets.Layout(width='150px'))
        elif param_type == "bool":
            return widgets.Checkbox(value=default, layout=widgets.Layout(width='150px'))
        elif param_type == "list_float":
            return widgets.Text(value=str(default), layout=widgets.Layout(width='150px'))
        elif param_type == "dropdown":
            return widgets.Dropdown(options=default, layout=widgets.Layout(width='150px'))
        else:
            return widgets.Text(value=str(default), layout=widgets.Layout(width='150px'))

    def update_param_inputs(self, *args):
        plot_key = self.plot_type.value
        spec = self.plot_specs.get(plot_key, {})
        params = {**self.default_params, **spec.get("params", {})}
        self.param_inputs.clear()
        widgets_list = []
        for name, meta in params.items():
            widget = self._create_widget(meta["type"], meta["default"])
            label = widgets.Label(value=f"{name}:", layout=widgets.Layout(width='100px'))
            hbox = widgets.HBox([label, widget])
            widgets_list.append(hbox)
            self.param_inputs[name] = widget
        rows = [widgets.HBox(widgets_list[i:i+2]) for i in range(0, len(widgets_list), 2)]
        self.param_box.children = rows

    def update_table_params(self):
        widgets_list = []
        for name, meta in self.default_params.items():
            widget = self._create_widget(meta["type"], meta["default"])
            label = widgets.Label(value=f"{name}:", layout=widgets.Layout(width='100px'))
            hbox = widgets.HBox([label, widget])
            widgets_list.append(hbox)
            self.param_inputs[name] = widget
        rows = [widgets.HBox(widgets_list[i:i+2]) for i in range(0, len(widgets_list), 2)]
        self.table_param_box.children = rows

    def do_plot(self, _):
        with self.output:
            clear_output(wait=True)
            subject = self.subject_dropdown.value
            task, run = self.task_dropdown.value
            kwargs = {
                k: (eval(w.value) if self.plot_specs[self.plot_type.value]["params"].get(k, {}).get("type") == "list_float" else w.value)
                for k, w in self.param_inputs.items()
            }
            spec = self.plot_specs[self.plot_type.value]
            filtered = extract_params({"params": {**self.default_params, **spec["params"]}}, kwargs)
            spec["function"](subject, task, run, **filtered)
            self.update_param_inputs()

    def do_show_info(self, _):
        with self.output:
            clear_output(wait=True)
            subject = self.subject_dropdown.value
            task, run = self.task_dropdown.value
            l_freq = float(self.param_inputs["l_freq"].value)
            h_freq = float(self.param_inputs["h_freq"].value)

            metadata = self.controller.show_annotations(subject, task, run)
            print(f"Metadata for {subject} - {task}" + (f" (Run {run})" if run else "") + ":")
            print(json.dumps(metadata, indent=2) if metadata else "No metadata available.")

            table_name = self.table_type.value
            rows = self.rows_int.value
            df = self.controller.show_table(subject, task, run, name=table_name, l_freq=l_freq, h_freq=h_freq, rows=rows)
            print(f"\nTable: {table_name}")
            if df is not None:
                display(df)
            else:
                print("No table data available.")

    def show(self):
        display(self.ui)
