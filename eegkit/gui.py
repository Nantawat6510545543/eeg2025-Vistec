import ipywidgets as widgets
from IPython.display import display, clear_output
import json
from .controller import EEGController

def parse_time_input(text_value):
    text_value = text_value.strip()
    return None if text_value == "" or text_value.lower() == "none" else float(text_value)

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

        self.plot_type = widgets.ToggleButtons(
            options=['time', 'sensors', 'frequency', 'conditionwise psd', 'epochs', 'evoked'],
            description='Plot:',
            layout=widgets.Layout(width='600px')
        )
        self.stimulus_dropdown = widgets.Dropdown(
            options=[],
            description='Stimulus:',
            layout=widgets.Layout(width='250px')
        )

        # Filter and PSD options
        self.lfreq_float = widgets.FloatText(value=3.0, description='l_freq:', layout=widgets.Layout(width='200px'))
        self.hfreq_float = widgets.FloatText(value=35.0, description='h_freq:', layout=widgets.Layout(width='200px'))
        self.average_check = widgets.Checkbox(value=True, description='Average', indent=False)
        self.db_check = widgets.Checkbox(value=True, description='dB', indent=False)

        # Time-domain controls
        self.duration_float = widgets.FloatText(value=10.0, description='duration:', layout=widgets.Layout(width='200px'))
        self.start_float = widgets.FloatText(value=0.0, description='start:', layout=widgets.Layout(width='200px'))
        self.nchan_int = widgets.IntText(value=10, description='n_channels:', layout=widgets.Layout(width='200px'))

        # Epoch cropping and frequency bounds
        self.tmin_text = widgets.Text(value="1.0", description='tmin:', layout=widgets.Layout(width='200px'))
        self.tmax_text = widgets.Text(value="2.4", description='tmax:', layout=widgets.Layout(width='200px'))
        self.fmin_float = widgets.FloatText(value=1.0, description='fmin:', layout=widgets.Layout(width='200px'))
        self.fmax_float = widgets.FloatText(value=50.0, description='fmax:', layout=widgets.Layout(width='200px'))

        self.plot_button = widgets.Button(description='Plot', button_style='success')
        self.info_button = widgets.Button(description='Show Info', button_style='info')

        self.table_type = widgets.Dropdown(
            options=['events', 'channels', 'electrodes', 'epochs'],
            description='Table:',
            layout=widgets.Layout(width='250px')
        )
        self.rows_int = widgets.IntText(
            value=10,
            description='Rows:',
            layout=widgets.Layout(width='200px')
        )

        self.output = widgets.Output()

        # Containers
        self.filter_controls = widgets.HBox([self.lfreq_float, self.hfreq_float])
        self.time_controls = widgets.HBox([self.duration_float, self.start_float, self.nchan_int])
        self.psd_options = widgets.HBox([self.average_check, self.db_check])
        self.t_controls = widgets.HBox([self.tmin_text, self.tmax_text])
        self.f_controls = widgets.HBox([self.fmin_float, self.fmax_float])
        self.param_box = widgets.VBox([])
        self.table_controls = widgets.HBox([self.table_type, self.rows_int, self.info_button])

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


    def update_tasks(self, *args):
        subject = self.subject_dropdown.value
        task_keys = sorted(self.controller.list_tasks(subject))
        formatted = [(f"{t} (run {r})" if r else t, (t, r)) for t, r in task_keys]
        self.task_dropdown.options = formatted
        if formatted:
            self.task_dropdown.value = formatted[0][1]
            self.update_stimulus_options()

    def update_param_inputs(self, *args):
        plot_mode = self.plot_type.value
        if plot_mode == 'conditionwise psd':
            self.param_box.children = [self.t_controls, self.f_controls, self.filter_controls, self.psd_options]
        elif plot_mode == 'frequency':
            self.param_box.children = [self.f_controls, self.filter_controls, self.psd_options]
        elif plot_mode == 'time':
            self.param_box.children = [self.time_controls, self.filter_controls]
        elif plot_mode == 'epochs':
            self.param_box.children = [self.nchan_int,self.t_controls, self.filter_controls, self.stimulus_dropdown]
            self.update_stimulus_options()
        elif plot_mode == 'evoked':
            self.param_box.children = [self.t_controls, self.filter_controls, self.stimulus_dropdown]
            self.update_stimulus_options()
        else:
            self.param_box.children = []

    def update_stimulus_options(self):
        subject = self.subject_dropdown.value
        task, run = self.task_dropdown.value
        run = run if run else None       
        l_freq = self.lfreq_float.value
        h_freq = self.hfreq_float.value

        event_ids = self.controller.get_event_ids(subject, task, l_freq, h_freq, run)
        self.stimulus_dropdown.options = sorted(event_ids)

    def do_plot(self, _):
        with self.output:
            clear_output(wait=True)
            subject = self.subject_dropdown.value
            task, run = self.task_dropdown.value

            tmin = parse_time_input(self.tmin_text.value)
            tmax = parse_time_input(self.tmax_text.value)

            kwargs = {
                'tmin': tmin,
                'tmax': tmax,
                'fmin': self.fmin_float.value,
                'fmax': self.fmax_float.value,
                'l_freq': self.lfreq_float.value,
                'h_freq': self.hfreq_float.value,
                'duration': self.duration_float.value,
                'start': self.start_float.value,
                'n_channels': self.nchan_int.value,
                'average': self.average_check.value,
                'dB': self.db_check.value,
                'stimulus': self.stimulus_dropdown.value,
            }
            self.controller.show(subject, task, run, plot_type=self.plot_type.value, **kwargs)

    def do_show_info(self, _):
        with self.output:
            clear_output(wait=True)
            subject = self.subject_dropdown.value
            task, run = self.task_dropdown.value
            l_freq = self.lfreq_float.value
            h_freq = self.hfreq_float.value

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