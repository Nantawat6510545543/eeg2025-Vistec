import pandas as pd
from ..views.visualization import EEGVisualization 

class EEGController:
    def __init__(self, subject_model, plot_spec_path):
        self.subject_model = subject_model
        self.visualizer = EEGVisualization(
            get_raw_func=self.get_filtered_raw,
            get_epochs_func=self.get_epochs,
            json_path=plot_spec_path
        )

    def get_filtered_raw(self, subject, task, run, l_freq, h_freq):
        return self.subject_model.get_task(subject, task, run).get_filtered_raw(l_freq, h_freq)

    def get_epochs(self, subject, task, run, l_freq, h_freq):
        return self.subject_model.get_task(subject, task, run).get_epochs(l_freq, h_freq)

    def list_subjects(self):
        return self.subject_model.list_subjects()

    def list_tasks(self, subject):
        return self.subject_model.list_tasks(subject)

    def get_event_ids(self, subject, task, l_freq, h_freq, run=None):
        task_model = self.subject_model.get_task(subject, task, run)
        epochs, _ = task_model.get_epochs(l_freq, h_freq)
        return list(epochs.event_id.keys()) if epochs else []

    def get_plot_specs(self):
        return self.visualizer.plot_specs

    def get_default_params(self):
        return self.visualizer.default_params

    def show(self, subject, task, run=None, plot_type='time', **kwargs):
        spec = self.visualizer.plot_specs.get(plot_type)
        if spec:
            return spec["function"](subject, task, run, **kwargs)
        else:
            print(f"Plot type '{plot_type}' is not defined.")

    def show_annotations(self, subject, task, run=None):
        task_model = self.subject_model.get_task(subject, task, run)
        return task_model.show_annotations()

    def show_table(self, subject, task, run=None, name='events', rows=10, l_freq=1, h_freq=50):
        task_model = self.subject_model.get_task(subject, task, run)
        return task_model.show_table(name=name, rows=rows, l_freq=l_freq, h_freq=h_freq)

    def get_annotation_df(self, subject, task, run=None):
        task_model = self.subject_model.get_task(subject, task, run)
        raw = task_model.get_filtered_raw(l_freq=1, h_freq=50)
        annots = raw.annotations
        df = pd.DataFrame({
            "onset": annots.onset,
            "duration": annots.duration,
            "description": annots.description
        })
        return df
