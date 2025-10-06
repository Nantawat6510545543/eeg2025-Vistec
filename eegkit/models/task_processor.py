import numpy as np
import mne
from mne import Epochs, events_from_annotations
from .dtos import BaseTaskDTO, FilterParamsDTO, EpochParamsDTO, EvokedParamsDTO
from ..cache import CacheKey
import logging
from ..utils import EEGCleaner

mne.set_log_level('WARNING')
import itertools

# fixed mapping
BACKGROUND = [0, 1]
FOREGROUND = [0.0, 0.3, 0.6, 1.0]
STIM = [1, 2, 3]
EVENT_ID = {
    f"bg{b}_fg{f:.1f}_stim{s}": i + 1
    for i, (b, f, s) in enumerate(itertools.product(BACKGROUND, FOREGROUND, STIM))
}

RESTING_STATE_EVENT_ID = {
    'open': 1,
    'close': 2,
}

CCD_EVENT_ID = {
    'trial_start': 1,
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

        self.preprocessors = _PREPROCESSORS
        self._log = logging.getLogger(__name__)

    def get_filtered(self, params: FilterParamsDTO):
        # 1) Find cleaned cache
        clean_ck = CacheKey(
            subject=self.task_dto.subject,
            task=self.task_dto.task,
            run=self.task_dto.run,
            stage="cleaned",
            params=params.cleaning_key,
            pipeline_ver=self.cache.pipeline_ver,
        )
        cleaned_cached = self.cache.load_raw_filtered(clean_ck)
        if cleaned_cached is not None:
            return cleaned_cached

        # 2) If cleaned cache not found, Build prefilter cache (Bandpass/resample/notch)
        pre_ck = CacheKey(
            subject=self.task_dto.subject,
            task=self.task_dto.task,
            run=self.task_dto.run,
            stage="prefilter",
            params=params.filter_key,
            pipeline_ver=self.cache.pipeline_ver,
        )
        cached = self.cache.load_raw_filtered(pre_ck)
        if cached is None:
            raw_pref = EEGCleaner.pre_filter(self.get_raw(), params)
            p = self.cache.save_raw_filtered(raw_pref, pre_ck)
            del raw_pref
            raw_pref = mne.io.read_raw_fif(p.as_posix(), preload=False, verbose="ERROR")
        else:
            raw_pref = cached

        # 3) Mark bad channels/time windows and save cleaned cache
        try:
            raw_clean = EEGCleaner.clean_mark(raw_pref, params)
            self.cache.save_raw_filtered(raw_clean, clean_ck)
            return raw_clean
        except Exception:
            return raw_pref

    def _apply_stimulus_filter(self, epochs: Epochs, params: EpochParamsDTO):
        stim = params.stimulus
        if isinstance(stim, (list, tuple)):
            stim = stim[0] if len(stim) > 0 else None
        if stim:
            if stim in epochs.event_id:
                return epochs[stim]
            self._log.warning("stim '%s' not in available event IDs %s", stim, list(epochs.event_id.keys()))
            return None
        return epochs.apply_baseline(baseline=(None, 0.0))

    def get_epochs(self, params: EpochParamsDTO):
        preprocess_fn = self.preprocessors.get(self.task_dto.task)
        if preprocess_fn is None:
            self._log.warning("Unsupported task for epochs: '%s'", self.task_dto.task)
            return None, "unavailable"

        ck = CacheKey(
            subject=self.task_dto.subject,
            task=self.task_dto.task,
            run=self.task_dto.run,
            stage="epochs",
            params=params.epochs_key,
            pipeline_ver=self.cache.pipeline_ver,
        )
        epochs, labels = self.cache.load_epochs(ck)
        if epochs is not None:
            epochs_sel = self._apply_stimulus_filter(epochs, params)
            if epochs_sel is None:
                return None, "unavailable"
            return epochs_sel, labels

        epochs, labels = preprocess_fn(self, params)
        if epochs is None:
            return None, "unavailable"
        
        try:
            if epochs.info.get('bads'):
                epochs = epochs.interpolate_bads(reset_bads=True)
        except Exception:
            pass

        if self.cache and ck:
            self.cache.save_epochs(epochs, ck, labels=labels)

        epochs_sel = self._apply_stimulus_filter(epochs, params)
        if epochs_sel is None:
            return None, "unavailable"

        return epochs_sel, labels

    def get_evoked(self, params: EvokedParamsDTO):
        ck = CacheKey(
            subject=self.task_dto.subject,
            task=self.task_dto.task,
            run=self.task_dto.run,
            stage="evoked",
            params=params.evoked_key,
            pipeline_ver=self.cache.pipeline_ver,
        )

        evk = self.cache.load_evoked(ck)
        if evk is not None:
            return evk

        epochs, _labels = self.get_epochs(params)
        if epochs is None:
            return None

        try:
            if epochs.info.get('bads'):
                epochs = epochs.interpolate_bads(reset_bads=True)
        except Exception:
            pass

        evoked = epochs.average()
        if self.cache and ck:
            self.cache.save_evoked(evoked, ck)

        return evoked

    @register_preprocessor("surroundSupp")
    def _sus_preprocess(self, params: EpochParamsDTO):
        filtered = self.get_filtered(params)
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
            tmin=params.tmin, tmax=params.tmax, proj=True,
            preload=False
        )
        labels = stim_rows["label"].values[epochs.selection]
        return epochs, labels

    @register_preprocessor("RestingState")
    def _resting_preprocess(self, params: EpochParamsDTO):
        filtered = self.get_filtered(params)
        df = self.get_events()
        if df is None:
            return None, None
        try:
            t_start = df[df['value'] == 'resting_start']['onset'].values[0]
            t_end = df[df['value'] == 'break cnt']['onset'].values[1]
        except Exception:
            return None, None
        filtered.crop(tmin=t_start, tmax=t_end)

        events_arr, ann_event_id = events_from_annotations(self.get_raw())
        open_code = ann_event_id.get('instructed_toOpenEyes')
        close_code = ann_event_id.get('instructed_toCloseEyes')
        new_events_list = []
        present = []
        if open_code is not None:
            open_rows = events_arr[events_arr[:, 2] == open_code]
            for row in open_rows:
                new_events_list.append([row[0], 0, RESTING_STATE_EVENT_ID['open']])
            if open_rows.size > 0:
                present.append('open')
        if close_code is not None:
            close_rows = events_arr[events_arr[:, 2] == close_code]
            for row in close_rows:
                new_events_list.append([row[0], 0, RESTING_STATE_EVENT_ID['close']])
            if close_rows.size > 0:
                present.append('close')
        if not new_events_list:
            return None, None
        new_events = np.array(sorted(new_events_list, key=lambda r: r[0]), dtype=int)
        event_id_sub = {k: RESTING_STATE_EVENT_ID[k] for k in present}
        epochs = Epochs(
            filtered, new_events, event_id=event_id_sub,
            tmin=params.tmin, tmax=params.tmax,
            proj=True, preload=False
        )
        labels = list(event_id_sub.keys())
        return epochs, labels

    @register_preprocessor("contrastChangeDetection")
    def _ccd_preprocess(self, params: EpochParamsDTO):
        filtered = self.get_filtered(params)
        events_arr, ann_event_id = events_from_annotations(self.get_raw())
        if 'contrastTrial_start' not in ann_event_id:
            return None, None
        ccd_code = ann_event_id['contrastTrial_start']
        trial_rows = events_arr[events_arr[:, 2] == ccd_code]
        if trial_rows.size == 0:
            return None, None
        new_events = np.column_stack([
            trial_rows[:, 0].astype(int),
            np.zeros(trial_rows.shape[0], dtype=int),
            np.full(trial_rows.shape[0], CCD_EVENT_ID['trial_start'], dtype=int)
        ])
        epochs = Epochs(
            filtered, new_events, event_id=CCD_EVENT_ID,
            tmin=params.tmin, tmax=params.tmax,
            proj=True, preload=False
        )
        labels = ['trial_start']
        return epochs, labels
