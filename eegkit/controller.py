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
    
    def get_event_ids(self, subject, task,l_freq, h_freq, run=None):
        task_data = self.subject_data.get_task(subject, task, run)
        epochs, _ = task_data.get_epochs(l_freq=l_freq, h_freq=h_freq)
        return list(epochs.event_id.keys()) if epochs else []

    def show(self, subject, task, run=None, plot_type='time', **kwargs):
        if plot_type == 'time':
            self.visualizer.plot_time(subject, task, run, **kwargs)
        elif plot_type == 'sensors':
            self.visualizer.plot_sensors(subject, task, run)
        elif plot_type == 'frequency':
            self.visualizer.plot_frequency(subject, task, run)
        elif plot_type == 'conditionwise psd':
            self.visualizer.plot_conditionwise_psd(subject, task, run, **kwargs)
        elif plot_type == 'epochs':
            self.visualizer.plot_epochs_or_evoked(subject, task, run, mode='epochs', **kwargs)
        elif plot_type == 'evoked':
            self.visualizer.plot_epochs_or_evoked(subject, task, run, mode='evoked', **kwargs)

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