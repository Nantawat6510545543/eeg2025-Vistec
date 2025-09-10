import numpy as np
import mne
from mne import Epochs, events_from_annotations
from .dtos import BaseTaskDTO, FilterParamsDTO, EpochParamsDTO
from ..cache import CacheKey

mne.set_log_level('WARNING')
import itertools

BACKGROUND = [0, 1]
FOREGROUND = [0.0, 0.3, 0.6, 1.0]
STIM = [1, 2, 3]
EVENT_ID = {
    f"bg{b}_fg{f:.1f}_stim{s}": i + 1
    for i, (b, f, s) in enumerate(itertools.product(BACKGROUND, FOREGROUND, STIM))
}

_PREPROCESSORS = {}


def register_preprocessor(task_name: str):
    def _decorator(func):
        _PREPROCESSORS[task_name] = func
        return func

    return _decorator


class EEGTaskProcessor:

    def __init__(self, get_raw_fn, get_events_fn, task_dto: BaseTaskDTO, cache):
        self.get_raw = get_raw_fn
        self.get_events = get_events_fn
        self.task_dto = task_dto
        self.cache = cache

        self._filtered_cache = {}
        self._epochs_cache = {}
        self.preprocessors = _PREPROCESSORS

    def get_filtered(self, params: FilterParamsDTO, no_pick=False):
        key = (params.l_freq, params.h_freq)

        # --- Check memory cache ---
        if key in self._filtered_cache:
            raw_out = self._filtered_cache[key]
        else:
            # --- Check disk cache ---
            ck = CacheKey(
                subject=self.task_dto.subject,
                task=self.task_dto.task,
                run=self.task_dto.run,
                stage="rawfilt",
                params={"l_freq": params.l_freq, "h_freq": params.h_freq},
                pipeline_ver=self.cache.pipeline_ver,
            )
            cached = self.cache.load_raw_filtered(ck)
            if cached is not None:
                self._filtered_cache[key] = cached
                raw_out = cached
            else:
                raw_copy = self.get_raw().copy().load_data()
                raw_copy.filter(
                    l_freq=params.l_freq,
                    h_freq=params.h_freq,
                    fir_design="firwin",
                    skip_by_annotation="edge"
                )
                self._filtered_cache[key] = raw_copy
                raw_out = raw_copy
                self.cache.save_raw_filtered(raw_copy, ck)

        if no_pick:
            return raw_out.copy()
        channels = params.channels_list
        return raw_out.copy().pick(channels)

    def get_epochs(self, params: EpochParamsDTO):
        key = (params.l_freq, params.h_freq, params.tmin, params.tmax)
        channels = params.channels_list

        # --- Check memory cache ---
        if key in self._epochs_cache:
            epochs, labels = self._epochs_cache[key]
            return epochs.copy().pick(channels), labels

        ck = CacheKey(
            subject=self.task_dto.subject,
            task=self.task_dto.task,
            run=self.task_dto.run,
            stage="epochs",
            params={
                "l_freq": params.l_freq,
                "h_freq": params.h_freq,
                "tmin": params.tmin,
                "tmax": params.tmax,
            },
            pipeline_ver=self.cache.pipeline_ver,
        )
        epochs, labels = self.cache.load_epochs(ck)
        if epochs is not None:
            self._epochs_cache[key] = (epochs, labels)
            return epochs.copy().pick(channels), labels

        preprocess_fn = self.preprocessors.get(self.task_dto.task)
        if preprocess_fn is None:
            print(f"Unsupported task for epochs: '{self.task_dto.task}'")
            return None, "unavailable"

        epochs, labels = preprocess_fn(self, params)
        if epochs is None:
            return None, "unavailable"
        self._epochs_cache[key] = (epochs, labels)
        if self.cache and ck:
            self.cache.save_epochs(epochs, ck, labels=labels)

        return epochs.copy().pick(channels), labels

    @register_preprocessor("surroundSupp")
    def _sus_preprocess(self, params: EpochParamsDTO):
        filtered = self.get_filtered(params, no_pick=True)
        events = self.get_events()
        if events is None:
            return None, None
        stim_rows = events[events['value'] == 'stim_ON'].copy()
        if stim_rows.empty:
            return None, None

        stim_rows['label'] = stim_rows.apply(
            lambda row: f"bg{int(row['background'])}_fg{row['foreground_contrast']}_stim{int(row['stimulus_cond'])}",
            axis=1
        )
        stim_rows["event_code"] = stim_rows["label"].map(EVENT_ID)

        events_array = np.column_stack([
            stim_rows['sample'].astype(int),
            np.zeros(len(stim_rows), dtype=int),
            stim_rows['event_code'].astype(int)
        ])

        present_labels = stim_rows["label"].unique()
        event_id_sub = {k: EVENT_ID[k] for k in present_labels}

        epochs = Epochs(
            filtered, events=events_array, event_id=event_id_sub,
            tmin=params.tmin, tmax=params.tmax, baseline=None, proj=True,
            preload=True, detrend=1
        )
        labels = stim_rows["label"].values[epochs.selection]
        return epochs, labels

    @register_preprocessor("RestingState")
    def _resting_preprocess(self, params: EpochParamsDTO):
        filtered = self.get_filtered(params, no_pick=True)
        df = self.get_events()
        if df is None:
            return None, None
        try:
            t_start = df[df['value'] == 'resting_start']['onset'].values[0]
            t_end = df[df['value'] == 'break cnt']['onset'].values[1]
        except Exception:
            return None, None
        filtered.crop(tmin=t_start, tmax=t_end)

        events, event_id = events_from_annotations(self.get_raw())
        eye_map = {
            'open': event_id.get('instructed_toOpenEyes'),
            'close': event_id.get('instructed_toCloseEyes')
        }
        eye_map = {k: v for k, v in eye_map.items() if v is not None}
        if not eye_map:
            return None, None

        epochs = Epochs(
            filtered, events, event_id=eye_map,
            tmin=params.tmin, tmax=params.tmax, baseline=None,
            proj=True, preload=True
        )
        labels = list(eye_map.keys())
        return epochs, labels

    @register_preprocessor("contrastChangeDetection")
    def _ccd_preprocess(self, params: EpochParamsDTO):
        filtered = self.get_filtered(params, no_pick=True)
        events, event_id = events_from_annotations(self.get_raw())
        if 'contrastTrial_start' not in event_id:
            return None, None
        ccd_code = event_id['contrastTrial_start']
        epochs = Epochs(
            filtered, events, event_id={'trial_start': ccd_code},
            tmin=params.tmin, tmax=params.tmax, baseline=None,
            proj=True, preload=True
        )
        labels = ['trial_start']
        return epochs, labels
