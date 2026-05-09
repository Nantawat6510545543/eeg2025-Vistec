"""Signal processing pipeline: filtering, epoching and evoked computation with caching."""

import logging

import mne
import numpy as np
import pandas as pd
from mne import Epochs, events_from_annotations

from .constants import (
    EVENT_ID,
    RESTING_STATE_EVENT_ID,
    CCD_EVENT_ID,
)
from .label_builders import (
    build_ccd_trial_outcome_labels,
    CCD_TARGET_HIT,
    CCD_TARGET_WRONG,
    CCD_TARGET_MISS,
)
from ..dtos import TaskDTO, FilterParamsDTO, EpochParamsDTO, EvokedParamsDTO
from ...cache import CacheKey
from ...utils.signal import EEGCleaner

mne.set_log_level('WARNING')

# Constants now sourced from .constants

_PREPROCESSORS = {}


def register_preprocessor(task_name: str):
    """Register a preprocessor function for a given task name."""

    def _decorator(func):
        _PREPROCESSORS[task_name] = func
        return func

    return _decorator


class EEGTaskProcessor:
    """Apply preprocessing recipe per task and manage cached intermediate artifacts."""

    def __init__(self, get_raw_fn, get_events_fn, task_dto: TaskDTO, cache):
        """Bind raw/events accessors, DTO and cache handle."""
        self.get_raw = get_raw_fn
        self.get_events = get_events_fn
        self.task_dto = task_dto
        self.cache = cache

        self.preprocessors = _PREPROCESSORS
        self._log = logging.getLogger(__name__)

    # Simpler non-cached version (
    def get_filtered(self, params: FilterParamsDTO):
        """Return cleaned Raw."""
        raw_pref = EEGCleaner.pre_filter(self.get_raw(), params)
        raw_clean = EEGCleaner.clean_mark(raw_pref, params)
        return raw_clean
    
    # Cache version
    # def get_filtered(self, params: FilterParamsDTO):
    #     """Return cleaned Raw using cached prefilter/clean stages when available."""
    #     # 1) Find cleaned cache
    #     clean_ck = CacheKey(
    #         subject=self.task_dto.subject,
    #         task=self.task_dto.task,
    #         run=self.task_dto.run,
    #         stage="cleaned",
    #         params=params.cleaning_key,
    #         pipeline_ver=self.cache.pipeline_ver,
    #     )
    #     cleaned_cached = self.cache.load_raw_filtered(clean_ck)
    #     if cleaned_cached is not None:
    #         return cleaned_cached

    #     # 2) If cleaned cache not found, Build prefilter cache (Bandpass/resample/notch)
    #     pre_ck = CacheKey(
    #         subject=self.task_dto.subject,
    #         task=self.task_dto.task,
    #         run=self.task_dto.run,
    #         stage="prefilter",
    #         params=params.filter_key,
    #         pipeline_ver=self.cache.pipeline_ver,
    #     )
    #     cached = self.cache.load_raw_filtered(pre_ck)
    #     if cached is None:
    #         raw_pref = EEGCleaner.pre_filter(self.get_raw(), params)
    #         p = self.cache.save_raw_filtered(raw_pref, pre_ck)
    #         del raw_pref
    #         raw_pref = mne.io.read_raw_fif(p.as_posix(), preload=False, verbose="ERROR")
    #     else:
    #         raw_pref = cached

    #     # 3) Mark bad channels/time windows and save cleaned cache
    #     raw_clean = EEGCleaner.clean_mark(raw_pref, params)
    #     self.cache.save_raw_filtered(raw_clean, clean_ck)
    #     return raw_clean

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
        """Return (epochs, labels) via registered task preprocessor with stimulus filter."""
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

        if epochs.info.get('bads'):
            epochs = epochs.interpolate_bads(reset_bads=True)

        if self.cache and ck:
            self.cache.save_epochs(epochs, ck, labels=labels)

        epochs_sel = self._apply_stimulus_filter(epochs, params)
        if epochs_sel is None:
            return None, "unavailable"

        return epochs_sel, labels

    def get_evoked(self, params: EvokedParamsDTO):
        """Return evoked average from epochs, caching on disk when possible."""
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

        if epochs.info.get('bads'):
            epochs = epochs.interpolate_bads(reset_bads=True)

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

        # Build human-readable labels and numeric event codes
        stim_rows['label'] = stim_rows.apply(
            lambda row: f"bg{int(row['background'])}_fg{row['foreground_contrast']}_stim{int(row['stimulus_cond'])}",
            axis=1
        )
        stim_rows["event_code"] = stim_rows["label"].map(EVENT_ID)

        # Derive sample indices aligned to the CURRENT Raw sampling rate
        sfreq = float(filtered.info.get('sfreq', 0.0))
        if 'onset' in stim_rows.columns:
            # Prefer onset (seconds) -> samples at current sfreq
            stim_rows = stim_rows[stim_rows['onset'].notna()].copy()
            stim_rows['sample'] = np.round(stim_rows['onset'].astype(float) * sfreq).astype(int)
        elif 'sample' in stim_rows.columns:
            # Fallback: use provided sample index (assume already at current rate)
            stim_rows['sample'] = stim_rows['sample'].astype(int)
        else:
            # Cannot construct events without timing
            return None, None

        # Guard against out-of-bounds epochs (edges of the recording)
        n_times = int(filtered.n_times)
        tmin_samp = int(np.floor(params.tmin * sfreq))
        tmax_samp = int(np.ceil(params.tmax * sfreq))
        start_idx = stim_rows['sample'] + tmin_samp
        stop_idx = stim_rows['sample'] + tmax_samp
        in_bounds = (start_idx >= 0) & (stop_idx < n_times)
        if not np.any(in_bounds):
            return None, None
        stim_rows = stim_rows.loc[in_bounds].copy()

        # Ensure events are sorted by sample index
        stim_rows.sort_values('sample', inplace=True)

        events_array = np.column_stack([
            stim_rows['sample'].astype(int).values,
            np.zeros(len(stim_rows), dtype=int),
            stim_rows['event_code'].astype(int).values
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

        starts = df[df.get('value') == 'resting_start'].get('onset') if 'value' in df and 'onset' in df else None
        ends = df[df.get('value').isin(['break cnt', 'resting_end'])][
            'onset'] if 'value' in df and 'onset' in df else None
        if starts is not None and len(starts) > 0 and ends is not None and len(ends) > 0:
            t_start = float(starts.iloc[0])
            t_end = float(ends.iloc[-1])
            if t_end > t_start:
                # Guard against events-derived onsets that slightly exceed the true
                # recording boundary due to rounding / resampling.
                max_t = float(filtered.times[-1]) if getattr(filtered, "n_times", 0) else None
                if max_t is not None:
                    if t_end > max_t:
                        self._log.warning(
                            "RestingState crop t_end (%.5f) exceeds max time (%.5f); clamping",
                            t_end,
                            max_t,
                        )
                    t_end = min(t_end, max_t)
                    t_start = max(0.0, min(t_start, max_t))

                if t_end > t_start:
                    try:
                        filtered.crop(tmin=t_start, tmax=t_end)
                    except ValueError as e:
                        # Final safety: clamp just inside the boundary.
                        if max_t is not None:
                            sfreq = float(filtered.info.get("sfreq", 0.0) or 0.0)
                            eps = (1.0 / sfreq) if sfreq > 0 else 1e-3
                            t_end_safe = max(t_start, max_t - eps)
                            if t_end_safe > t_start:
                                self._log.warning(
                                    "RestingState crop failed (%s); retry with tmax=%.5f",
                                    str(e),
                                    t_end_safe,
                                )
                                filtered.crop(tmin=t_start, tmax=t_end_safe)
                            else:
                                self._log.warning("RestingState crop failed (%s); skipping crop", str(e))
                        else:
                            raise

        # Compute events from annotations on the Raw
        events_arr, ann_event_id = events_from_annotations(filtered)
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
        events_df = self.get_events()
        # Build one outcome label per trial_start window from the parsed event table.
        trial_outcomes = build_ccd_trial_outcome_labels(events_df)

        raw = self.get_raw()
        events_arr, ann_event_id = events_from_annotations(raw)
        if 'contrastTrial_start' not in ann_event_id:
            return None, None
        ccd_code = ann_event_id['contrastTrial_start']
        trial_rows = events_arr[events_arr[:, 2] == ccd_code]
        if trial_rows.size == 0:
            return None, None

        if len(trial_outcomes) != int(trial_rows.shape[0]):
            raise ValueError(
                "CCD trial mismatch before epoching: "
                f"{int(trial_rows.shape[0])} trial_start annotations vs {len(trial_outcomes)} outcome labels"
            )

        # Strictly validate trial order alignment between events_df and raw annotations.
        # This prevents silent label rotation when both sources have equal length but different ordering.
        if events_df is None or events_df.empty or "onset" not in events_df.columns or "value" not in events_df.columns:
            raise ValueError("CCD events_df must contain non-empty onset/value columns for strict alignment")

        df_align = events_df.copy()
        df_align["onset"] = pd.to_numeric(df_align["onset"], errors="coerce")
        df_align = df_align[np.isfinite(df_align["onset"])].copy()
        value_norm = np.asarray([str(v).strip().lower() for v in df_align["value"].to_numpy()], dtype=object)
        df_trial = df_align[value_norm == "contrasttrial_start"].sort_values("onset")
        trial_onsets_df = df_trial["onset"].to_numpy(dtype=float)

        raw_sfreq = float(raw.info.get('sfreq', 0.0))
        if raw_sfreq <= 0:
            raise ValueError("Invalid raw sampling rate for CCD alignment")
        trial_onsets_raw = trial_rows[:, 0].astype(float) / raw_sfreq

        if len(trial_onsets_df) != len(trial_onsets_raw):
            raise ValueError(
                "CCD trial-start onset mismatch: "
                f"events_df has {len(trial_onsets_df)} starts vs raw has {len(trial_onsets_raw)}"
            )

        # One-sample tolerance at raw sampling grid.
        tol_sec = max(1.0 / raw_sfreq, 1e-3)
        onset_delta = np.abs(trial_onsets_df - trial_onsets_raw)
        bad_idx = np.where(onset_delta > tol_sec)[0]
        if bad_idx.size > 0:
            i = int(bad_idx[0])
            raise ValueError(
                "CCD trial order/alignment mismatch at index "
                f"{i}: events_df onset={trial_onsets_df[i]:.6f}, raw onset={trial_onsets_raw[i]:.6f}, "
                f"delta={onset_delta[i]:.6f}s > tol={tol_sec:.6f}s"
            )

        # Drop trial anchors that would create out-of-bounds epochs at the requested tmin/tmax.
        samples = trial_rows[:, 0].astype(int)
        sfreq = float(filtered.info.get('sfreq', 0.0))
        n_times = int(filtered.n_times)
        tmin_samp = int(np.floor(params.tmin * sfreq))
        tmax_samp = int(np.ceil(params.tmax * sfreq))
        start_idx = samples + tmin_samp
        stop_idx = samples + tmax_samp
        in_bounds = (start_idx >= 0) & (stop_idx < n_times)
        if not np.any(in_bounds):
            return None, None

        kept_samples = samples[in_bounds]
        kept_outcomes = np.asarray(trial_outcomes, dtype=object)[in_bounds]

        outcome_to_code = {
            CCD_TARGET_HIT: CCD_EVENT_ID['hit'],
            CCD_TARGET_WRONG: CCD_EVENT_ID['wrong'],
            CCD_TARGET_MISS: CCD_EVENT_ID['miss'],
        }
        norm_outcomes = [str(v).strip().lower() for v in kept_outcomes.tolist()]
        invalid = sorted({v for v in norm_outcomes if v not in outcome_to_code})
        if invalid:
            raise ValueError(f"Unsupported CCD outcomes: {invalid}")

        # Encode per-epoch class directly in events[:, 2] so epochs['hit'|'wrong'|'miss'] works.
        event_codes = np.asarray([outcome_to_code[v] for v in norm_outcomes], dtype=int)
        new_events = np.column_stack([
            kept_samples,
            np.zeros(len(kept_samples), dtype=int),
            event_codes,
        ])

        present = set(norm_outcomes)
        event_id_sub = {
            k: CCD_EVENT_ID[k]
            for k in [CCD_TARGET_HIT, CCD_TARGET_WRONG, CCD_TARGET_MISS]
            if k in present
        }

        epochs = Epochs(
            filtered, new_events, event_id=event_id_sub,
            tmin=params.tmin, tmax=params.tmax,
            proj=True, preload=False
        )

        # Return unique available conditions for UI stimulus options.
        labels = list(event_id_sub.keys())
        return epochs, labels
