import numpy as np
from mne import Epochs, events_from_annotations
from .dtos import FilterParamsDTO, EpochParamsDTO

class EEGTaskProcessor:
    def __init__(self, raw, events, task_name: str):
        self.raw = raw
        self.events = events
        self.task_name = task_name
        self._filtered_cache = {}
        self._epochs_cache = {}

    def get_filtered(self, params: FilterParamsDTO):
        key = (params.l_freq, params.h_freq)

        if key in self._filtered_cache:
            return self._filtered_cache[key]
        raw_copy = self.raw.copy().load_data()

        raw_copy.filter(
            l_freq=params.l_freq,
            h_freq=params.h_freq,
            fir_design="firwin",
            skip_by_annotation="edge"
        )

        self._filtered_cache[key] = raw_copy
        return raw_copy

    def get_epochs(self, params: EpochParamsDTO):
        key = (params.l_freq, params.h_freq, params.tmin, params.tmax)
        if key in self._epochs_cache:
            return self._epochs_cache[key]

        if self.task_name == 'RestingState':
            epochs, labels = self._resting_preprocess(params)
        elif self.task_name == 'surroundSupp':
            epochs, labels = self._sus_preprocess(params)
        elif self.task_name == 'contrastChangeDetection':
            epochs, labels = self._ccd_preprocess(params)
        else:
            print("epochs fail with " + self.task_name)
            return None, None

        self._epochs_cache[key] = (epochs, labels)
        return epochs, labels

    def _sus_preprocess(self, params: EpochParamsDTO):
        filtered = self.get_filtered(params)
        stim_rows = self.events[self.events['value'] == 'stim_ON'].copy()

        stim_rows['label'] = stim_rows.apply(
            lambda row: f"bg{int(row['background'])}_fg{row['foreground_contrast']}_stim{int(row['stimulus_cond'])}",
            axis=1
        )
        labels = stim_rows['label'].unique()
        event_id = {label: idx + 1 for idx, label in enumerate(sorted(labels))}
        stim_rows['event_code'] = stim_rows['label'].map(event_id)

        events_array = np.column_stack([
            stim_rows['sample'].astype(int),
            np.zeros(len(stim_rows), dtype=int),
            stim_rows['event_code'].astype(int)
        ])
        epochs = Epochs(
            filtered, events=events_array, event_id=event_id,
            tmin=params.tmin, tmax=params.tmax, baseline=None, proj=True,
            preload=True, detrend=1
        )
        labels = stim_rows['label'].values[epochs.selection]
        return epochs, labels

    def _resting_preprocess(self, params: EpochParamsDTO):
        filtered = self.get_filtered(params)
        df = self.events
        t_start = df[df['value'] == 'resting_start']['onset'].values[0]
        t_end = df[df['value'] == 'break cnt']['onset'].values[1]
        filtered.crop(tmin=t_start, tmax=t_end)

        events, event_id = events_from_annotations(self.raw)
        eye_map = {
            'open': event_id['instructed_toOpenEyes'],
            'close': event_id['instructed_toCloseEyes']
        }

        epochs = Epochs(
            filtered, events, event_id=eye_map,
            tmin=params.tmin, tmax=params.tmax, baseline=None,
            proj=True, preload=True
        )
        labels = eye_map
        return epochs, labels

    def _ccd_preprocess(self, params: EpochParamsDTO):
        filtered = self.get_filtered(params)
        events, event_id = events_from_annotations(self.raw)
        ccd_code = event_id['contrastTrial_start']
        epochs = Epochs(
            filtered, events, event_id={'trial_start': ccd_code},
            tmin=params.tmin, tmax=params.tmax, baseline=None,
            proj=True, preload=True
        )
        labels = ['trial_start']
        return epochs, labels
