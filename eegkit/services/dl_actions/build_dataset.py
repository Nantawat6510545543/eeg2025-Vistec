"""Build a raw epoch tensor dataset (shape N, 1, C, T) for DL training."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ...models.dtos import DLEpochDatasetParamsDTO, BaseTaskDTO
from ...utils.channels import prepare_channels
from . import register_dl


def _build_ccd_trial_labels(events_df: pd.DataFrame) -> np.ndarray:
    """Return one binary label per CCD trial (1=hit, 0=miss) from a raw events DataFrame."""
    if events_df is None or events_df.empty:
        return np.asarray([], dtype=np.int64)
    if "onset" not in events_df.columns or "value" not in events_df.columns:
        return np.asarray([], dtype=np.int64)

    df = events_df.copy()
    df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
    df = df.dropna(subset=["onset"]).sort_values("onset").reset_index(drop=True)

    trial_starts = np.where(df["value"].eq("contrastTrial_start").values)[0]
    if trial_starts.size == 0:
        return np.asarray([], dtype=np.int64)

    onsets = df["onset"].to_numpy()
    values = df["value"].astype(str).to_numpy()
    labels: List[int] = []

    for i, s_idx in enumerate(trial_starts):
        t_start = float(onsets[s_idx])
        t_end = float(onsets[trial_starts[i + 1]]) if i + 1 < len(trial_starts) else np.inf
        in_win = (onsets > t_start) & (onsets < t_end)
        press_mask = np.isin(values, ["left_buttonPress", "right_buttonPress"]) & in_win
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


def _build_subject_chunk(task_model, params: DLEpochDatasetParamsDTO) -> Dict:
    """Build x (N,1,C,T), y, group arrays for a single task model."""
    epochs, _labels = task_model.get_epochs(params)
    if epochs is None:
        return {
            "x": np.empty((0, 1, 0, 0), dtype=np.float32),
            "y": np.empty((0,), dtype=np.int64),
            "group": np.empty((0,), dtype=object),
        }

    epochs = prepare_channels(epochs, params)
    x_data = epochs.get_data(copy=True).astype(np.float32)   # (N, C, T)
    x = x_data[:, np.newaxis, :, :]                           # (N, 1, C, T)

    events_df = task_model.get_event() if hasattr(task_model, "get_event") else pd.DataFrame()
    y = _build_ccd_trial_labels(events_df if events_df is not None else pd.DataFrame())

    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]

    subject = getattr(getattr(task_model, "task_dto", None), "subject", None)
    group = np.asarray([subject if subject is not None else "unknown"] * n, dtype=object)
    return {"x": x, "y": y, "group": group}


@register_dl("Build DL Dataset", DLEpochDatasetParamsDTO)
def build_dataset(self, task_dto: BaseTaskDTO, params: DLEpochDatasetParamsDTO):
    """Build epoch tensor dataset and return {name, x:(N,1,C,T), y:(N,), group:(N,)}."""
    task_model = self.get_task(task_dto) if self.get_task is not None else None
    if task_model is None:
        return {
            "name": params.dataset_name,
            "x": np.empty((0, 1, 0, 0), dtype=np.float32),
            "y": np.empty((0,), dtype=np.int64),
            "group": np.empty((0,), dtype=object),
        }

    task_models = list(getattr(task_model, "task_model_list", [])) or [task_model]
    chunks = []
    for single_model in tqdm(task_models, desc="Build DL dataset subjects", leave=False):
        chunk = _build_subject_chunk(single_model, params)
        if len(chunk["y"]) > 0:
            chunks.append(chunk)

    if not chunks:
        return {
            "name": params.dataset_name,
            "x": np.empty((0, 1, 0, 0), dtype=np.float32),
            "y": np.empty((0,), dtype=np.int64),
            "group": np.empty((0,), dtype=object),
        }

    return {
        "name": params.dataset_name,
        "x": np.concatenate([c["x"] for c in chunks]),
        "y": np.concatenate([c["y"] for c in chunks]),
        "group": np.concatenate([c["group"] for c in chunks]),
    }

