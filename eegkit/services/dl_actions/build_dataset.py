"""Build a raw epoch tensor dataset (shape N, 1, C, T) for DL training."""

from __future__ import annotations

from typing import Dict

import numpy as np
from tqdm.auto import tqdm

from ...models.dtos import DLEpochDatasetParamsDTO, BaseTaskDTO
from ...models.pipeline.label_builders import CCD_TARGET_HIT, CCD_TARGET_WRONG, CCD_TARGET_MISS
from ...utils.channels import prepare_channels
from . import register_dl


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

