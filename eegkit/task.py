import mne
from mne import events_from_annotations, Epochs
import json
import pandas as pd
import numpy as np

class EEGTaskData:
    def __init__(self, subject, task, run, data_dir):
        self.subject = subject
        self.task = task
        self.run = run
        self.data_dir = data_dir

        self._raw = None
        self._filtered_cache = {}  # key = (l_freq, h_freq)
        self.metadata = {}
        self.events = None
        self.channels = None
        self.electrodes = None

        self._epochs_cache = {}  # key: (l_freq, h_freq) → (epochs, labels)

        self._load()


    def _get_file(self, ext):
        base = f"{self.subject}_task-{self.task}"
        if self.run:
            base += f"_run-{self.run}"
        file_name = f"{base}_{ext}"

        return self.data_dir / f"{self.subject}" / "eeg" / file_name

    def _load(self):
        eeg_path = self._get_file("eeg.set")
        self._raw = mne.io.read_raw_eeglab(eeg_path, preload=True, montage_units='cm')
        montage = mne.channels.make_standard_montage("GSN-HydroCel-128")
        self._raw.drop_channels(['Cz'])
        self._raw.set_montage(montage, match_case=False)
        # self._raw.filter(l_freq=self.l_freq, h_freq=self.h_freq)

        json_path = self._get_file("eeg.json")
        if json_path.exists():
            with open(json_path) as f:
                self.metadata = json.load(f)

        event_path = self._get_file("events.tsv")
        if event_path.exists():
            self.events = pd.read_csv(event_path, sep='\t')

        channels_path = self._get_file("channels.tsv")
        if channels_path.exists():
            self.channels = pd.read_csv(channels_path, sep='\t')

        electrodes_path = self._get_file("electrodes.tsv")
        if electrodes_path.exists():
            self.electrodes = pd.read_csv(electrodes_path, sep='\t')

    def get_filtered_raw(self, l_freq=1, h_freq=50):
        key = (l_freq, h_freq)
        
        # Return cached version if available
        if key in self._filtered_cache:
            return self._filtered_cache[key]

        # Filter and cache
        raw_copy = self._raw.copy().load_data()
        raw_copy.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin", skip_by_annotation="edge")
        
        self._filtered_cache[key] = raw_copy
        return raw_copy

    def get_epochs(self, l_freq=1, h_freq=50):
        key = (l_freq, h_freq)
        if key in self._epochs_cache:
            return self._epochs_cache[key]

        if self.task == 'RestingState':
            epochs, labels = self._resting_preprocess(l_freq, h_freq)
        elif self.task == 'surroundSupp':
            epochs, labels = self._Sus_preprocess(l_freq, h_freq)
        else:
            return None, None  # Unsupported task

        if epochs is not None:
            self._epochs_cache[key] = (epochs, labels)

        return epochs, labels

    def _Sus_preprocess(self, tmin=0.0, duration=2.4, l_freq=1, h_freq=50):
        """
        Preprocess surroundSupp task using 'stim_ON' events.
        Epochs are 2.4s long and labeled by background + foreground_contrast + stimulus_cond.
        """
        filtered_raw = self.get_filtered_raw(l_freq=l_freq, h_freq=h_freq)
        df = self.events
        stim_rows = df[df['value'] == 'stim_ON'].copy()

        stim_rows['label'] = stim_rows.apply(
            lambda row: f"bg{int(row['background'])}_fg{row['foreground_contrast']}_stim{int(row['stimulus_cond'])}",
            axis=1
        )

        # Create event_id mapping from existing unique labels
        unique_labels = sorted(stim_rows['label'].unique())
        event_id = {label: idx + 1 for idx, label in enumerate(unique_labels)}
        stim_rows['event_code'] = stim_rows['label'].map(event_id)

        # Build events array
        events_array = np.column_stack([
            stim_rows['sample'].astype(int),
            np.zeros(len(stim_rows), dtype=int),
            stim_rows['event_code'].astype(int)
        ])

        # Epoching
        tmax = tmin + duration
        epochs = Epochs(
            filtered_raw,
            events=events_array,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=None,
            proj=True,
            preload=True,
            detrend=1
        )
        labels = stim_rows['label'].values
        labels = labels[epochs.selection]
        return epochs, labels

    def _resting_preprocess(self, tmin=0.0, tmax=20.0, l_freq=1, h_freq=50):
        """
        Crop raw based on 'resting_start' to 'break cnt' in events.tsv,
        then epoch using eye condition annotations.
        """
        filtered_raw = self.get_filtered_raw(l_freq=l_freq, h_freq=h_freq)

        # Step 1: Find resting_start and break cnt from TSV
        df = self.events

        t_start = df[df['value'] == 'resting_start']['onset'].values[0]
        t_end = df[df['value'] == 'break cnt']['onset'].values[1]

        # Step 2: Crop raw to this resting window
        filtered_raw.crop(tmin=t_start, tmax=t_end)

        # Step 3: Extract new events from cropped raw's annotations
        events, event_id = events_from_annotations(self._raw)

        eye_event_id = {
            'open': event_id['instructed_toOpenEyes'],
            'close': event_id['instructed_toCloseEyes']
        }

        # Step 4: Create epochs based on eye condition labels
        epochs = Epochs(
            filtered_raw,
            events=events,
            event_id=eye_event_id,
            tmin=tmin,
            tmax=tmax,
            proj=True,
            baseline=None,
            preload=True
        )

        labels = epochs.events[:, -1] - eye_event_id['open']  # 0=open, 1=close
        
        return epochs, labels

    def show_annotations(self):
        return self.metadata if self.metadata else None

    def show_table(self, name='events', rows=10, l_freq=1, h_freq=50):
        df_map = {
            'events': self.events,
            'channels': self.channels,
            'electrodes': self.electrodes
        }

        if name == 'epochs':
            epochs, labels = self.get_epochs(l_freq=l_freq, h_freq=h_freq)
            if epochs is None:
                return None

            info = {
                'n_epochs': len(epochs),
                'n_channels': len(epochs.ch_names),
                'timespan_sec': epochs.times[-1] - epochs.times[0],
                'labels': np.unique(labels) if labels is not None else 'N/A',
                'sampling_rate': epochs.info['sfreq'],
                'duration_per_epoch_sec': epochs.get_data().shape[-1] / epochs.info['sfreq']
            }
            return pd.DataFrame([info])

        df = df_map.get(name)
        return df.head(rows) if df is not None else None

    def get_raw(self):
        return self._raw

