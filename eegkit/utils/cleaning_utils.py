"""EEG cleaning helpers inspired by EEGLAB Clean Rawdata.

Class-based API `EEGCleaner` performs a complete preprocessing pipeline on MNE Raw objects:
- Band-pass filter (l_freq/h_freq), resample, and notch filter
- Mark flatline channels as bad (> N seconds constant)
- Mark high-noise channels (HF band z-score > threshold) as bad
- Mark low-correlation channels (vs. robust reference) as bad
- Mark bad time windows where a large fraction of channels are out of power range

All operations are optional and controlled via FilterParamsDTO clean_* fields.
"""
from __future__ import annotations

import logging
import time
import numpy as np
import mne
import asrpy
from numpy.lib.stride_tricks import sliding_window_view as swv

log = logging.getLogger(__name__)


def _safe_zscore(x: np.ndarray, axis=None):
    mu = np.nanmean(x, axis=axis, keepdims=True)
    sd = np.nanstd(x, axis=axis, keepdims=True)
    sd = np.where(sd == 0, np.nan, sd)
    return (x - mu) / sd


class EEGCleaner:
    """Stateless cleaning utilities as static methods.

    Public API:
    - pre_filter(raw, params) -> Raw: band-pass, resample, notch (in-place on the passed Raw), returns Raw
    - clean_mark(raw, params) -> Raw: mark bad channels and bad time windows, returns Raw
    """

    @staticmethod
    def _mark_bad_flatline_channels(raw, params):
        t0 = time.perf_counter()
        flat_sec = getattr(params, 'clean_flatline_sec', None)
        if flat_sec is None or flat_sec <= 0:
            log.info("[flatline] disabled (clean_flatline_sec=%s)", flat_sec)
            return raw
        sfreq = float(raw.info.get('sfreq', 0.0)) or 0.0
        n_samples = int(round(flat_sec * sfreq))
        if n_samples <= 1:
            log.info("[flatline] window too small (n=%d) -> skip", n_samples)
            return raw
        picks = mne.pick_types(raw.info, eeg=True)
        if picks.size == 0:
            log.info("[flatline] no EEG picks -> skip")
            return raw
        data = raw.get_data(picks=picks, reject_by_annotation='omit')
        bad_names = []
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
            log.info("[flatline] Marking %d channels as bad (>%ss): %s", len(bad_names), flat_sec, bad_names)
            raw = raw.copy()
            raw.info['bads'] = sorted(set((raw.info.get('bads') or []) + bad_names))
        else:
            log.info("[flatline] no channels flagged (%.3fs)", time.perf_counter() - t0)
        return raw

    @staticmethod
    def _mark_bad_highfreq_noise_channels(raw, params):
        t0 = time.perf_counter()
        hf_sd_max = getattr(params, 'clean_hf_noise_sd_max', None)
        if hf_sd_max is None or hf_sd_max <= 0:
            log.info("[hf-noise] disabled (clean_hf_noise_sd_max=%s)", hf_sd_max)
            return raw
        picks = mne.pick_types(raw.info, eeg=True)
        if picks.size == 0:
            log.info("[hf-noise] no EEG picks -> skip")
            return raw
        f_low_default, f_high_default = 30.0, 100.0
        hp = float(raw.info.get('highpass') or 0.0)
        lp = raw.info.get('lowpass')
        lp = float(lp) if (lp is not None and np.isfinite(lp)) else np.inf
        fmin = max(f_low_default, hp)
        fmax = min(f_high_default, lp)
        if not np.isfinite(fmax) or fmax <= fmin + 1.0:
            log.info("[hf-noise] insufficient passband (%.2f-%.2f Hz) -> skip", fmin, fmax)
            return raw
        try:
            spec = raw.compute_psd(method='welch', fmin=fmin, fmax=fmax, picks=picks,
                                   n_overlap=0, reject_by_annotation=True, verbose='ERROR')
            psd = spec.get_data()  # (n_ch, n_freqs)
            freqs = spec.freqs
            bandpower = np.trapz(psd, freqs, axis=-1)  # (n_ch,)
        except Exception as e:
            log.info("[hf-noise] PSD failed (%s) -> fallback to RMS of current band", e)
            data = raw.get_data(picks=picks, reject_by_annotation='omit')
            bandpower = np.sqrt(np.nanmean(data ** 2, axis=-1))
        z = _safe_zscore(bandpower)
        bad_local_idx = np.where(z.flatten() > hf_sd_max)[0].tolist()
        bad_names = [raw.ch_names[picks[i]] for i in bad_local_idx]
        if bad_names:
            log.info("[hf-noise] Marking %d channels bad (z>%.2f): %s", len(bad_names), hf_sd_max, bad_names)
            raw = raw.copy()
            raw.info['bads'] = sorted(set((raw.info.get('bads') or []) + bad_names))
        else:
            log.info("[hf-noise] no channels flagged (%.3fs)", time.perf_counter() - t0)
        return raw

    @staticmethod
    def _mark_bad_lowcorr_channels(raw, params):
        t0 = time.perf_counter()
        corr_min = getattr(params, 'clean_corr_min', None)
        if corr_min is None or not (0 < float(corr_min) <= 1):
            log.info("[lowcorr] disabled (clean_corr_min=%s)", corr_min)
            return raw
        picks = mne.pick_types(raw.info, eeg=True)
        if picks.size < 3:
            log.info("[lowcorr] <3 EEG channels -> skip")
            return raw
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
            log.info("[lowcorr] Marking %d channels bad (corr<%.2f): %s", len(bad_names), corr_min, bad_names)
            raw = raw.copy()
            raw.info['bads'] = sorted(set((raw.info.get('bads') or []) + bad_names))
        else:
            log.info("[lowcorr] no channels flagged (%.3fs)", time.perf_counter() - t0)
        return raw

    @staticmethod
    def _mark_bad_windows_by_power(raw, params):
        t0 = time.perf_counter()
        min_sd = getattr(params, 'clean_power_min_sd', float('-inf'))
        max_sd = getattr(params, 'clean_power_max_sd', None)
        max_out_pct = getattr(params, 'clean_max_outbound_pct', None)
        window_sec = getattr(params, 'clean_window_sec', None)
        if window_sec is None or window_sec <= 0:
            log.info("[power-win] disabled (window_sec=%s)", window_sec)
            return raw
        sfreq = float(raw.info.get('sfreq', 0.0))
        win = max(1, int(round(window_sec * sfreq)))
        picks = mne.pick_types(raw.info, eeg=True)
        if picks.size == 0:
            log.info("[power-win] no EEG picks -> skip")
            return raw
        data = raw.get_data(picks=picks, reject_by_annotation='omit')
        n_ch, n_t = data.shape
        if n_t < win:
            return raw
        v = swv(data, win, axis=1)
        power = np.nanmean(v ** 2, axis=-1)
        z = _safe_zscore(power, axis=1)
        # Handle None -> infinite bounds
        lower_bound = -np.inf if min_sd is None else float(min_sd)
        upper_bound = np.inf if max_sd is None else float(max_sd)
        lower_ok = z >= lower_bound
        upper_ok = z <= upper_bound
        ok = lower_ok & upper_ok
        frac_bad = 100.0 * (1.0 - np.nanmean(ok, axis=0))
        thr_pct = 25.0 if max_out_pct is None else float(max_out_pct)
        bad_windows = np.where(frac_bad > thr_pct)[0]
        if bad_windows.size == 0:
            log.info("[power-win] no windows flagged (%.3fs)", time.perf_counter() - t0)
            return raw
        onsets = bad_windows.astype(float) / sfreq
        durations = np.full_like(onsets, fill_value=win / sfreq, dtype=float)
        desc = ['bad_power'] * len(onsets)
        ann = mne.Annotations(onset=onsets.tolist(), duration=durations.tolist(), description=desc)
        raw = raw.copy()
        raw.set_annotations(raw.annotations + ann)
        log.info("[power-win] Annotated %d windows (min_sd=%s, max_sd=%s, max_out_pct=%s, win=%.3fs) in %.3fs",
                 len(onsets), min_sd, max_sd, max_out_pct, win / sfreq, time.perf_counter() - t0)
        return raw

    @staticmethod
    def _apply_asr(raw, params):
        t0 = time.perf_counter()
        window_sec = getattr(params, 'clean_window_sec', None)
        max_std = getattr(params, 'clean_asr_max_std', None)
        remove_only = getattr(params, 'clean_asr_remove_only', False)
        if max_std is None or max_std <= 0:
            log.info("[ASR] disabled (clean_asr_max_std=%s)", max_std)
            return raw
        try:
            sfreq = float(raw.info.get('sfreq', 0.0))
            picks = mne.pick_types(raw.info, eeg=True)
            if picks.size == 0:
                log.info("[ASR] no EEG picks -> skip")
                return raw
            data_uv = raw.get_data(picks=picks) * 1e6

            # Initialize ASR model; forward window length when supported by API.
            try:
                model = asrpy.ASR(sfreq=sfreq, cutoff=max_std, window_length=window_sec)
            except TypeError:
                model = asrpy.ASR(sfreq=sfreq, cutoff=max_std)
                # Best-effort: some versions expose attribute to set window length
                if window_sec and hasattr(model, 'window_length'):
                    try:
                        setattr(model, 'window_length', window_sec)
                    except Exception:
                        pass

            cleaned_uv = model.fit_transform(data_uv)

            # If remove_only, annotate windows with large residuals instead of reconstructing
            if remove_only:
                win = max(1, int(round(float(window_sec or 0.5) * sfreq)))
                if cleaned_uv.shape[1] >= win:
                    resid = data_uv - cleaned_uv  # microvolts
                    try:
                        v = swv(resid, win, axis=1)  # (n_ch, n_win, win)
                        rms_ch_win = np.sqrt(np.nanmean(v ** 2, axis=-1))  # (n_ch, n_win)
                        rms_win = np.nanmedian(rms_ch_win, axis=0)  # (n_win,)
                        z = _safe_zscore(rms_win, axis=None).flatten()
                        bad_windows = np.where(z > float(max_std))[0]
                    except Exception:
                        bad_windows = np.array([], dtype=int)
                    if bad_windows.size > 0:
                        onsets = bad_windows.astype(float) / sfreq
                        durations = np.full_like(onsets, fill_value=win / sfreq, dtype=float)
                        ann = mne.Annotations(
                            onset=onsets.tolist(), duration=durations.tolist(),
                            description=['bad_asr'] * len(onsets)
                        )
                        out = raw.copy()
                        out.set_annotations(out.annotations + ann)
                        log.info("[ASR] remove_only: annotated %d windows (win=%.3fs, cutoff=%.2f) in %.3fs",
                                 len(onsets), win / sfreq, float(max_std), time.perf_counter() - t0)
                        return out
                # If we get here, we couldn't compute a mask; fall back to original raw
                log.info("[ASR] remove_only: no windows flagged (%.3fs)", time.perf_counter() - t0)
                return raw

            # Default: replace data with ASR-cleaned output
            out = raw.copy()
            out._data[picks, :] = cleaned_uv / 1e6
            log.info("[ASR] reconstructed data for %d channels (win=%.3fs, cutoff=%.2f) in %.3fs",
                     len(picks), float(window_sec or 0.5), float(max_std), time.perf_counter() - t0)
            return out
        except Exception as e:
            log.warning("[ASR] failed (%s); skipping.", e)
            return raw

    # ---------- public API ----------
    @staticmethod
    def pre_filter(raw, params) -> mne.io.BaseRaw:
        """Apply band-pass, resample, and notch to self.raw in place (copy), using self.params."""
        t0 = time.perf_counter()
        l_freq = getattr(params, 'l_freq', 0.5)
        h_freq = getattr(params, 'h_freq', 55.0)
        resample_fs = getattr(params, 'resample_fs', 500.0)
        notch = getattr(params, 'notch', 60.0)
        log.info("[prefilter] start l=%.3f, h=%.3f, fs=%s, notch=%s", float(l_freq), float(h_freq), resample_fs, notch)
        raw.load_data()
        raw.filter(
            l_freq=l_freq,
            h_freq=h_freq,
            fir_design='firwin',
            skip_by_annotation='edge',
        )
        target_fs = float(getattr(params, 'resample_fs', 500.0) or 0.0)
        cur_fs = float(raw.info.get('sfreq', 0.0) or 0.0)
        if target_fs > 0 and abs(cur_fs - target_fs) > 1e-6:
            raw.resample(target_fs)
        notch = getattr(params, 'notch', 60.0)
        if notch and float(notch) > 0:
            raw.notch_filter(
                freqs=notch,
                fir_design='firwin',
                skip_by_annotation='edge',
            )
        log.info("[prefilter] done: sfreq=%.3f (t=%.3fs)", float(raw.info.get('sfreq', 0.0)), time.perf_counter() - t0)
        return raw

    @staticmethod
    def clean_mark(raw, params) -> mne.io.BaseRaw:
        """Only apply marking steps (bad channels and bad windows) to self.raw, without prefilter."""
        t0 = time.perf_counter()
        log.info("[clean_mark] start")

        raw = EEGCleaner._mark_bad_flatline_channels(raw, params)
        raw = EEGCleaner._mark_bad_highfreq_noise_channels(raw, params)
        raw = EEGCleaner._mark_bad_lowcorr_channels(raw, params)
        raw = EEGCleaner._apply_asr(raw, params)
        raw = EEGCleaner._mark_bad_windows_by_power(raw, params)
        log.info("[clean_mark] done (t=%.3fs)", time.perf_counter() - t0)
        return raw
