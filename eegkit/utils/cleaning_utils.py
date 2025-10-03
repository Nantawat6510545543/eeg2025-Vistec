"""EEG cleaning helpers inspired by EEGLAB Clean Rawdata.

Class-based API `EEGCleaner` performs a complete preprocessing pipeline on MNE Raw objects:
- Band-pass filter (l_freq/h_freq), resample, and notch filter
- Drop flatline channels (> N seconds constant)
- Drop high-noise channels (HF band z-score > threshold)
- Drop low-correlation channels (vs. robust reference)
- Mark/remove bad time windows where a large fraction of channels are out of power range

ASR is attempted via `asrpy` if installed and enabled; otherwise skipped.

All operations are optional and controlled via FilterParamsDTO clean_* fields.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import mne
import asrpy  

log = logging.getLogger(__name__)


def _safe_zscore(x: np.ndarray, axis=None):
    mu = np.nanmean(x, axis=axis, keepdims=True)
    sd = np.nanstd(x, axis=axis, keepdims=True)
    sd = np.where(sd == 0, np.nan, sd)
    return (x - mu) / sd


class EEGCleaner:
    """Class-based cleaner encapsulating preprocessing and cleaning steps.

    Typical usage:
        cleaner = EEGCleaner(params, raw)
        raw_clean = cleaner.clean_raw()

    Backward-compatible usage still works:
        EEGCleaner(params).clean(raw)
    """

    def __init__(self, params, raw: mne.io.BaseRaw):
        self.params = params
        self.raw = raw

    def _drop_flatline_channels(self):
        raw = self.raw
        flat_sec = getattr(self.params, 'clean_flatline_sec', 5.0)
        if not (flat_sec and flat_sec > 0):
            return
        sfreq = float(raw.info.get('sfreq', 0.0)) or 0.0
        n_samples = int(round(flat_sec * sfreq))
        if n_samples <= 1:
            return
        picks = mne.pick_types(raw.info, eeg=True)
        if picks.size == 0:
            return
        data = raw.get_data(picks=picks, reject_by_annotation='omit')
        bad_names = []
        from numpy.lib.stride_tricks import sliding_window_view as swv
        for idx, ch_idx in enumerate(picks):
            x = data[idx]
            if x.shape[-1] < n_samples:
                continue
            try:
                v = swv(x, n_samples)
                is_flat = np.any(np.nanmax(v, axis=-1) - np.nanmin(v, axis=-1) == 0)
            except Exception:
                is_flat = np.nanmax(x) - np.nanmin(x) == 0
            if is_flat:
                bad_names.append(raw.ch_names[ch_idx])
        if bad_names:
            log.info("Dropping flatline channels (>%ss): %s", flat_sec, bad_names)
            self.raw = raw.copy().drop_channels(bad_names)
        else:
            self.raw = raw

    def _drop_highfreq_noise_channels(self):
        raw = self.raw
        hf_sd_max = getattr(self.params, 'clean_hf_noise_sd_max', 4.0)
        if not (hf_sd_max and hf_sd_max > 0):
            return
        picks = mne.pick_types(raw.info, eeg=True)
        if picks.size == 0:
            return
        tmp = raw.copy().filter(l_freq=30.0, h_freq=100.0, fir_design='firwin', picks=picks, verbose='ERROR')
        data = tmp.get_data(picks=picks, reject_by_annotation='omit')
        rms = np.sqrt(np.nanmean(data ** 2, axis=-1))
        z = _safe_zscore(rms)
        bad_local_idx = np.where(z.flatten() > hf_sd_max)[0].tolist()
        bad_names = [raw.ch_names[picks[i]] for i in bad_local_idx]
        if bad_names:
            log.info("Dropping high-frequency noise channels (z>%.2f): %s", hf_sd_max, bad_names)
            self.raw = raw.copy().drop_channels(bad_names)
        else:
            self.raw = raw
        try:
            tmp.close()
        except Exception:
            pass

    def _drop_lowcorr_channels(self):
        raw = self.raw
        corr_min = getattr(self.params, 'clean_corr_min', 0.8)
        if not (raw and corr_min and 0 < corr_min <= 1):
            return
        picks = mne.pick_types(raw.info, eeg=True)
        if picks.size < 3:
            return
        data = raw.get_data(picks=picks, reject_by_annotation='omit')
        ref = np.nanmedian(data, axis=0)
        corrs = []
        for i in range(data.shape[0]):
            x = data[i]
            mask = np.isfinite(x) & np.isfinite(ref)
            if mask.sum() < 10:
                corrs.append(1.0)
                continue
            cx = np.corrcoef(x[mask], ref[mask])[0, 1]
            corrs.append(abs(float(cx)))
        corrs = np.array(corrs)
        bad_local_idx = np.where(corrs < corr_min)[0].tolist()
        bad_names = [raw.ch_names[picks[i]] for i in bad_local_idx]
        if bad_names:
            log.info("Dropping low-correlation channels (corr<%.2f): %s", corr_min, bad_names)
            self.raw = raw.copy().drop_channels(bad_names)
        else:
            self.raw = raw

    def _mark_bad_windows_by_power(self):
        raw = self.raw
        min_sd = getattr(self.params, 'clean_power_min_sd', float('-inf'))
        max_sd = getattr(self.params, 'clean_power_max_sd', 7.0)
        max_out_pct = getattr(self.params, 'clean_max_outbound_pct', 25.0)
        window_sec = getattr(self.params, 'clean_window_sec', 0.5)
        if not (raw and window_sec and window_sec > 0):
            return
        sfreq = float(raw.info.get('sfreq', 0.0))
        win = max(1, int(round(window_sec * sfreq)))
        picks = mne.pick_types(raw.info, eeg=True)
        if picks.size == 0:
            return
        data = raw.get_data(picks=picks, reject_by_annotation='omit')
        n_ch, n_t = data.shape
        if n_t < win:
            return
        from numpy.lib.stride_tricks import sliding_window_view as swv
        v = swv(data, win, axis=1)
        power = np.nanmean(v ** 2, axis=-1)
        z = _safe_zscore(power, axis=1)
        lower_ok = True if np.isneginf(min_sd) else (z >= min_sd)
        upper_ok = z <= max_sd
        ok = lower_ok & upper_ok
        frac_bad = 100.0 * (1.0 - np.nanmean(ok, axis=0))
        bad_windows = np.where(frac_bad > float(max_out_pct))[0]
        if bad_windows.size == 0:
            return
        onsets = (bad_windows / sfreq) * win
        durations = np.full_like(onsets, fill_value=win / sfreq, dtype=float)
        desc = ['bad_power'] * len(onsets)
        ann = mne.Annotations(onset=onsets.tolist(), duration=durations.tolist(), description=desc)
        raw = raw.copy()
        raw.set_annotations(raw.annotations + ann)
        self.raw = raw

    def _apply_asr_if_available(self) -> None:
        raw = self.raw
        window_sec = getattr(self.params, 'clean_window_sec', 0.5)
        max_std = getattr(self.params, 'clean_asr_max_std', 20.0)
        remove_only = getattr(self.params, 'clean_asr_remove_only', True)
        if not (raw and max_std and max_std > 0):
            return
        try:
            sfreq = float(raw.info.get('sfreq', 0.0))
            picks = mne.pick_types(raw.info, eeg=True)
            if picks.size == 0:
                return
            data = raw.get_data(picks=picks) * 1e6
            model = asrpy.ASR(sfreq=sfreq, cutoff=max_std)
            cleaned = model.fit_transform(data)
            out = raw.copy()
            out._data[picks, :] = cleaned / 1e6
            self.raw = out
        except Exception as e:
            log.warning("ASR failed (%s); skipping.", e)
            self.raw = raw

    def _pre_filter(self) -> None:
        """Apply band-pass, resample, and notch to self.raw in place (copy), using self.params."""
        params = self.params
        out = self.raw.copy()
        try:
            out.load_data()
            out.filter(
                l_freq=getattr(params, 'l_freq', 0.5),
                h_freq=getattr(params, 'h_freq', 55.0),
                fir_design='firwin',
                skip_by_annotation='edge',
            )
            target_fs = float(getattr(params, 'resample_fs', 500.0) or 0.0)
            cur_fs = float(out.info.get('sfreq', 0.0) or 0.0)
            if target_fs > 0 and abs(cur_fs - target_fs) > 1e-6:
                out.resample(target_fs)
            notch = getattr(params, 'notch', 60.0)
            if notch and float(notch) > 0:
                out.notch_filter(
                    freqs=notch,
                    fir_design='firwin',
                    skip_by_annotation='edge',
                )
        except Exception as e:
            log.warning("Pre-clean filtering step failed: %s; proceeding with raw copy.", e)
        self.raw = out

    # ---------- public API ----------
    def clean_raw(self) -> mne.io.BaseRaw:
        """Run preprocessing and cleaning on self.raw using self.params.

        Returns the cleaned Raw. self.raw is updated to the cleaned copy.
        """
        params = self.params
        self._pre_filter()

        try:
            if getattr(params, 'clean_remove_bad_channels', False):
                self._drop_flatline_channels()
                self._drop_highfreq_noise_channels()
                self._drop_lowcorr_channels()

            if getattr(params, 'clean_asr', False):
                self._apply_asr_if_available()

            self._mark_bad_windows_by_power()
        except Exception as e:
            log.warning("Cleaning step failed: %s (skipped)", e)
            return self.raw
        return self.raw


def clean_raw_like_eeglab(raw: mne.io.BaseRaw, params) -> mne.io.BaseRaw:
    """Backwards-compatible function wrapper returning cleaned Raw."""
    return EEGCleaner(params, raw).clean_raw()
