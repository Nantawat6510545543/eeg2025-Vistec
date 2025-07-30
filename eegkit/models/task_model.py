from .task_loader import EEGTaskLoader
from .task_processor import EEGTaskProcessor
import pandas as pd

class EEGTaskModel:
    def __init__(self, subject, task, run, data_dir):
        self.loader = EEGTaskLoader(subject, task, run, data_dir)

        self.raw = self.loader.load_raw()
        self.events = self.loader.load_events()
        self.channels = self.loader.load_channels()
        self.electrodes = self.loader.load_electrodes()
        self.metadata = self.loader.load_metadata()

        self.processor = EEGTaskProcessor(self.raw, self.events, task)

    def get_filtered_raw(self, l_freq, h_freq):
        return self.processor.get_filtered(l_freq, h_freq)

    def get_epochs(self, l_freq, h_freq):
        return self.processor.get_epochs(l_freq, h_freq)

    def show_table(self, name='events', rows=10, l_freq=1, h_freq=50):
        if name == 'epochs':
            epochs, labels = self.get_epochs(l_freq, h_freq)
            if epochs is None:
                return None
            info = {
                'n_epochs': len(epochs),
                'n_channels': len(epochs.ch_names),
                'timespan_sec': epochs.times[-1] - epochs.times[0],
                'labels': sorted(set(labels)) if labels is not None else 'N/A',
                'sampling_rate': epochs.info['sfreq'],
                'duration_per_epoch_sec': epochs.get_data().shape[-1] / epochs.info['sfreq']
            }
            return pd.DataFrame([info])

        df_map = {
            'events': self.events,
            'channels': self.channels,
            'electrodes': self.electrodes
        }
        return df_map.get(name, pd.DataFrame()).head(rows)

    def show_annotations(self):
        return self.metadata if self.metadata else None
