from __future__ import annotations

from typing import Dict, List

import antropy as ent
import numpy as np
import pandas as pd
from scipy.signal import welch
from tqdm.auto import tqdm

from ...models.dtos import BaseTaskDTO, EpochParamsDTO
from ...utils.channels import prepare_channels
from . import register_ml


def build_ccd_trial_labels(events_df: pd.DataFrame) -> np.ndarray:
    """Build one CCD trial label per trial start: 1=hit, 0=miss/non-hit."""
    if events_df is None or events_df.empty:
        return np.asarray([], dtype=np.int64)

    if "onset" not in events_df.columns or "value" not in events_df.columns:
        return np.asarray([], dtype=np.int64)

    df = events_df.copy()
    df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
    df = df.dropna(subset=["onset"]).sort_values("onset").reset_index(drop=True)

    trial_start_idx = np.where(df["value"].eq("contrastTrial_start").values)[0]
    if trial_start_idx.size == 0:
        return np.asarray([], dtype=np.int64)

    labels: List[int] = []
    onsets = df["onset"].to_numpy()
    values = df["value"].astype(str).to_numpy()

    for i, start_idx in enumerate(trial_start_idx):
        t_start = float(onsets[start_idx])
        t_end = float(onsets[trial_start_idx[i + 1]]) if i + 1 < len(trial_start_idx) else np.inf

        in_window = (onsets > t_start) & (onsets < t_end)
        press_mask = np.isin(values, ["left_buttonPress", "right_buttonPress"]) & in_window
        press_idx = np.where(press_mask)[0]

        if press_idx.size == 0:
            labels.append(0)
            continue

        first_press = int(press_idx[0])
        feedback = None
        if "feedback" in df.columns and pd.notna(df.iloc[first_press].get("feedback")):
            feedback = str(df.iloc[first_press]["feedback"]).strip()

        labels.append(1 if feedback == "smiley_face" else 0)

    return np.asarray(labels, dtype=np.int64)


def _safe_hjorth(signal_1d: np.ndarray) -> Dict[str, float]:
    """Return Hjorth activity/mobility/complexity with zero-division guards."""
    activity = float(np.var(signal_1d))
    diff_1 = np.diff(signal_1d)
    var_diff_1 = float(np.var(diff_1)) if diff_1.size else 0.0
    mobility = float(np.sqrt(var_diff_1 / activity)) if activity > 0 else 0.0

    diff_2 = np.diff(diff_1)
    var_diff_2 = float(np.var(diff_2)) if diff_2.size else 0.0
    mobility_diff_1 = float(np.sqrt(var_diff_2 / var_diff_1)) if var_diff_1 > 0 else 0.0
    complexity = float(mobility_diff_1 / mobility) if mobility > 0 else 0.0

    return {
        "hjorth_activity": activity,
        "hjorth_mobility": mobility,
        "hjorth_complexity": complexity,
    }


def _extract_time_freq_features(signal_1d: np.ndarray, fs: float) -> Dict[str, float]:
    """Extract time-frequency features used to build ML vectors."""
    freqs, psd = welch(signal_1d, fs=fs, nperseg=min(len(signal_1d), 256))
    max_idx = int(np.argmax(psd)) if psd.size else 0

    feats = {
        "peak_to_peak": float(np.ptp(signal_1d)),
        "msv": float(np.mean(signal_1d ** 2)),
        "variance": float(np.var(signal_1d)),
        "max_psd_freq": float(freqs[max_idx]) if freqs.size else 0.0,
        "max_psd_value": float(np.max(psd)) if psd.size else 0.0,
        "power_sum": float(np.sum(psd)) if psd.size else 0.0,
    }
    feats.update(_safe_hjorth(signal_1d))
    return feats


def _extract_nonlinear_features(signal_1d: np.ndarray, fs: float) -> Dict[str, float]:
    """Extract nonlinear features using antropy."""
    hist, _ = np.histogram(signal_1d, bins=10, density=True)
    hist = hist[hist > 0]
    shannon_entropy = float(-np.sum(hist * np.log2(hist))) if hist.size else 0.0

    return {
        "approx_entropy": float(ent.app_entropy(signal_1d)),
        "sample_entropy": float(ent.sample_entropy(signal_1d)),
        "perm_entropy": float(ent.perm_entropy(signal_1d, normalize=True)),
        "svd_entropy": float(ent.svd_entropy(signal_1d, normalize=True)),
        "shannon_entropy": shannon_entropy,
        "spectral_entropy": float(ent.spectral_entropy(signal_1d, sf=fs, method="welch", normalize=True)),
    }


def _extract_epoch_vector(epoch_data: np.ndarray, fs: float) -> np.ndarray:
    """Extract a feature vector for one epoch by concatenating per-channel features."""
    row_features: List[float] = []
    for channel_signal in epoch_data:
        tf_feats = _extract_time_freq_features(channel_signal, fs)
        nl_feats = _extract_nonlinear_features(channel_signal, fs)
        feature_values = list(tf_feats.values()) + list(nl_feats.values())
        row_features.extend(float(v) for v in feature_values)
    return np.asarray(row_features, dtype=np.float64)


def _empty_dataset() -> Dict[str, np.ndarray]:
    """Return an empty dataset payload with stable dtypes."""
    return {
        "x": np.empty((0, 0), dtype=np.float64),
        "y": np.empty((0,), dtype=np.int64),
        "group": np.empty((0,), dtype=object),
    }


def _build_task_dataset(task_model, params: EpochParamsDTO) -> Dict[str, np.ndarray]:
    """Build one dataset chunk for a single task model/run."""
    epochs, _labels = task_model.get_epochs(params)
    if epochs is None:
        return _empty_dataset()

    epochs = prepare_channels(epochs, params)
    x_data = epochs.get_data(copy=True)
    sfreq = float(epochs.info.get("sfreq", 128.0))

    x_rows = []
    for epoch_data in tqdm(x_data, total=len(x_data), desc="Build dataset features", leave=False):
        x_rows.append(_extract_epoch_vector(epoch_data, sfreq))
    x = np.vstack(x_rows) if x_rows else np.empty((0, 0), dtype=np.float64)

    events_df = task_model.get_event() if hasattr(task_model, "get_event") else pd.DataFrame()
    y = build_ccd_trial_labels(events_df if events_df is not None else pd.DataFrame())

    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]

    subject = getattr(getattr(task_model, "task_dto", None), "subject", None)
    group = np.asarray([subject if subject is not None else "unknown"] * n, dtype=object)

    return {"x": x, "y": y, "group": group}


@register_ml("Build Dataset", EpochParamsDTO)
def build_dataset(self, task_dto: BaseTaskDTO, params: EpochParamsDTO):
    """Build ML-ready arrays from epochs and CCD trial labels."""
    task_model = self.get_task(task_dto) if self.get_task is not None else None
    if task_model is None:
        return _empty_dataset()

    if hasattr(task_model, "task_model_list"):
        task_models = list(getattr(task_model, "task_model_list", []))
    else:
        task_models = [task_model]

    chunks = []
    for single_task_model in tqdm(task_models, total=len(task_models), desc="Build dataset subjects", leave=False):
        chunk = _build_task_dataset(single_task_model, params)
        if len(chunk["y"]) > 0:
            chunks.append(chunk)

    if not chunks:
        return _empty_dataset()

    x = np.vstack([chunk["x"] for chunk in chunks])
    y = np.concatenate([chunk["y"] for chunk in chunks])
    group = np.concatenate([chunk["group"] for chunk in chunks])
    return {"x": x, "y": y, "group": group}
