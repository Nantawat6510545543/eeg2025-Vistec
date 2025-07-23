import pandas as pd
from .subject import EEGSubjectData
from .visualization import EEGVisualization

class EEGController:
    def __init__(self, subject_data: 'EEGSubjectData', visualizer: 'EEGVisualization'):
        self.subject_data = subject_data
        self.visualizer = visualizer

    def list_subjects(self):
        return self.subject_data.list_subjects()

    def list_tasks(self, subject):
        return self.subject_data.list_tasks(subject)

    def get_event_ids(self, subject, task, l_freq, h_freq, run=None):
        task_data = self.subject_data.get_task(subject, task, run)
        epochs, _ = task_data.get_epochs(l_freq=l_freq, h_freq=h_freq)
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
        """Return metadata dict or None."""
        task_data = self.subject_data.get_task(subject, task, run)
        return task_data.show_annotations() if task_data else None

    def show_table(self, subject, task, run=None, name='events', rows=10, l_freq=1, h_freq=50):
        """Return DataFrame or None."""
        task_data = self.subject_data.get_task(subject, task, run)
        return task_data.show_table(name=name, rows=rows, l_freq=l_freq, h_freq=h_freq)

    def get_annotation_df(self, subject, task, run=None):
        task_data = self.subject_data.get_task(subject, task, run)
        raw = task_data.get_filtered_raw()
        annots = raw.annotations
        df = pd.DataFrame({
            "onset": annots.onset,
            "duration": annots.duration,
            "description": annots.description
        })
        return df
