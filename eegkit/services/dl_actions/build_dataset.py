"""Build raw epoch tensor datasets (classification and regression) for DL training."""

from __future__ import annotations

import logging
from typing import Dict

import mne
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ...models.dtos import (
    BaseTaskDTO,
    DLEpochDatasetParamsDTO,
    DLEpochRegressionDatasetParamsDTO,
    DLEpochReactionTimeDatasetParamsDTO,
)
from ...models.pipeline.label_builders import CCD_TARGET_HIT, CCD_TARGET_WRONG, CCD_TARGET_MISS
from ...models.pipeline.label_builders import build_ccd_trial_outcomes_and_reaction_times
from ...utils.channels import prepare_channels
from . import register_dl


_LOG = logging.getLogger(__name__)


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

    if epochs.events is None or len(epochs.events) == 0:
        raise ValueError("CCD epochs contain no events for label mapping")

    outcome_to_binary = {
        CCD_TARGET_HIT: 1,
        CCD_TARGET_WRONG: 0,
        CCD_TARGET_MISS: 0,
    }

    id_to_label = {int(v): str(k).strip().lower() for k, v in epochs.event_id.items()}
    event_codes = np.asarray(epochs.events[:, 2], dtype=int)
    normalized = [id_to_label.get(int(code), "") for code in event_codes]
    invalid = sorted({v for v in normalized if v not in outcome_to_binary})
    if invalid:
        raise ValueError(f"Unsupported CCD event labels from preprocess: {invalid}")

    y = np.asarray([outcome_to_binary[v] for v in normalized], dtype=np.int64)

    if len(x) != len(y):
        raise ValueError(f"CCD x/y length mismatch: x={len(x)} epochs, y={len(y)} labels")

    n = len(x)

    subject = getattr(getattr(task_model, "task_dto", None), "subject", None)
    group = np.asarray([subject if subject is not None else "unknown"] * n, dtype=object)
    return {"x": x, "y": y, "group": group}


def _build_subject_reaction_time_chunk(task_model, params: DLEpochReactionTimeDatasetParamsDTO) -> Dict:
    """Build strict hit-only RT chunk with y as target-to-press latency in seconds."""
    empty_chunk = {
        "x": np.empty((0, 1, 0, 0), dtype=np.float32),
        "y": np.empty((0,), dtype=np.float32),
        "group": np.empty((0,), dtype=object),
    }

    epochs, _labels = task_model.get_epochs(params)
    if epochs is None:
        raise ValueError("RT dataset build failed: task_model.get_epochs(params) returned no epochs")

    task_dto = getattr(task_model, "task_dto", None)
    subject = getattr(task_dto, "subject", "unknown")
    run = getattr(task_dto, "run", "unknown")
    _LOG.info(
        "[rt-build] start subject=%s run=%s tmin=%.3f tmax=%.3f",
        subject,
        run,
        float(params.tmin),
        float(params.tmax),
    )

    events_df = task_model.get_event() if hasattr(task_model, "get_event") else None
    outcomes, reaction_times = build_ccd_trial_outcomes_and_reaction_times(events_df)
    if outcomes.size == 0 or reaction_times.size == 0:
        raise ValueError("RT dataset build failed: no CCD outcomes/reaction times were derived from events.tsv")

    # Use events.tsv only: get trial_start onsets (sec) and convert to epoch sample grid.
    if events_df is None or events_df.empty or "onset" not in events_df.columns or "value" not in events_df.columns:
        raise ValueError("RT dataset build failed: events.tsv must include onset/value columns")

    df_align = events_df.copy()
    df_align["onset"] = pd.to_numeric(df_align["onset"], errors="coerce")
    df_align = df_align[np.isfinite(df_align["onset"])].copy()
    value_norm = np.asarray([str(v).strip().lower() for v in df_align["value"].to_numpy()], dtype=object)
    df_trial = df_align[value_norm == "contrasttrial_start"].sort_values("onset")
    trial_onsets_sec = df_trial["onset"].to_numpy(dtype=float)
    if trial_onsets_sec.size == 0:
        raise ValueError("RT dataset build failed: no contrastTrial_start rows found in events.tsv")

    if int(trial_onsets_sec.size) != int(outcomes.size):
        raise ValueError(
            "RT dataset build failed: trial_start count mismatch between events.tsv and derived outcomes; "
            f"trial_starts={int(trial_onsets_sec.size)} outcomes={int(outcomes.size)}"
        )

    sfreq_epochs = float(epochs.info.get("sfreq", 0.0))
    if sfreq_epochs <= 0.0:
        raise ValueError("RT dataset build failed: invalid epochs sampling rate")

    trial_samples = np.rint(trial_onsets_sec * sfreq_epochs).astype(int)
    epoch_samples = np.asarray(epochs.events[:, 0], dtype=int)

    # Map epoch sample -> nearest trial sample (tolerance: 1 sample).
    tol_samp = 1
    pos = np.searchsorted(trial_samples, epoch_samples)
    pos0 = np.clip(pos, 0, trial_samples.size - 1)
    pos1 = np.clip(pos - 1, 0, trial_samples.size - 1)
    d0 = np.abs(trial_samples[pos0] - epoch_samples)
    d1 = np.abs(trial_samples[pos1] - epoch_samples)
    use1 = d1 < d0
    trial_pos = np.where(use1, pos1, pos0).astype(int)
    dist = np.where(use1, d1, d0)

    if np.any(dist > tol_samp):
        bad = np.flatnonzero(dist > tol_samp)
        i = int(bad[0])
        raise ValueError(
            "RT dataset build failed: could not map epoch samples back to trial_start samples within tolerance; "
            f"subject={subject} run={run} bad_epoch_idx={i} epoch_sample={int(epoch_samples[i])} "
            f"nearest_trial_sample={int(trial_samples[trial_pos[i]])} dist={int(dist[i])}"
        )

    epoch_outcomes = np.asarray(outcomes, dtype=object)[trial_pos]
    epoch_rts = np.asarray(reaction_times, dtype=np.float32)[trial_pos]

    epochs = prepare_channels(epochs, params)
    x_all = epochs.get_data(copy=True).astype(np.float32)[:, np.newaxis, :, :]
    hit_code = int(epochs.event_id.get(CCD_TARGET_HIT, -1))
    if hit_code < 0:
        _LOG.info("[rt-build] subject=%s run=%s skip: no hit code in epochs.event_id", subject, run)
        return empty_chunk

    hit_mask = np.asarray(epochs.events[:, 2], dtype=int) == hit_code
    selected_idx = np.flatnonzero(hit_mask)
    _LOG.info(
        "[rt-build] subject=%s run=%s hit_code=%d hit_epoch_count=%d hit_epoch_indices=%s",
        subject,
        run,
        hit_code,
        int(selected_idx.size),
        selected_idx.astype(int).tolist(),
    )

    if selected_idx.size == 0:
        _LOG.info("[rt-build] subject=%s run=%s skip: no hit epochs", subject, run)
        return empty_chunk

    y_all = epoch_rts[hit_mask]
    finite = np.isfinite(y_all)
    if not np.all(finite):
        _LOG.info(
            "[rt-build] subject=%s run=%s dropping %d/%d hit epochs due to non-finite RT",
            subject,
            run,
            int(np.sum(~finite)),
            int(finite.size),
        )
    selected_idx = selected_idx[finite]
    y = y_all[finite].astype(np.float32, copy=False)

    if selected_idx.size == 0:
        _LOG.info("[rt-build] subject=%s run=%s skip: no finite hit RT after filtering", subject, run)
        return empty_chunk

    x = x_all[selected_idx]
    _LOG.info("[rt-build] subject=%s run=%s emit_samples=%d", subject, run, int(len(y)))
    group = np.asarray([subject if subject is not None else "unknown"] * len(y), dtype=object)
    return {"x": x, "y": y, "group": group}


@register_dl("Build EEGNet Classification Dataset", DLEpochDatasetParamsDTO)
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
    for single_model in tqdm(task_models, desc="Build classification dataset subjects", leave=False):
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

@register_dl("Build EEGNet RT Dataset (Hit-Only)", DLEpochReactionTimeDatasetParamsDTO)
def build_reaction_time_dataset(self, task_dto: BaseTaskDTO, params: DLEpochReactionTimeDatasetParamsDTO):
    """Build strict hit-only reaction-time dataset for regression."""
    task_model = self.get_task(task_dto) if self.get_task is not None else None
    if task_model is None:
        return {
            "name": params.dataset_name,
            "x": np.empty((0, 1, 0, 0), dtype=np.float32),
            "y": np.empty((0,), dtype=np.float32),
            "group": np.empty((0,), dtype=object),
        }

    task_models = list(getattr(task_model, "task_model_list", [])) or [task_model]
    chunks = []
    for single_model in tqdm(task_models, desc="Build RT hit-only dataset subjects", leave=False):
        chunk = _build_subject_reaction_time_chunk(single_model, params)
        if len(chunk["y"]) > 0:
            chunks.append(chunk)

    if not chunks:
        return {
            "name": params.dataset_name,
            "x": np.empty((0, 1, 0, 0), dtype=np.float32),
            "y": np.empty((0,), dtype=np.float32),
            "group": np.empty((0,), dtype=object),
        }

    return {
        "name": params.dataset_name,
        "x": np.concatenate([c["x"] for c in chunks]),
        "y": np.concatenate([c["y"] for c in chunks]),
        "group": np.concatenate([c["group"] for c in chunks]),
    }

