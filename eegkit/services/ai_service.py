"""AI service: dataset building, simple training, and model registry introspection."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List

import numpy as np

from .base_service import BaseService
from ..utils.channels import prepare_channels
from ..models.dtos import (
    BaseTaskDTO,
    EpochParamsDTO,
    SubjectFilterDTO,
)

from .ai_actions import ai_registry  


class EEGAIService(BaseService):
    """Provide AI-related actions (list models, build dataset, train, predict placeholder)."""

    description = "AI training and inference on epochs (registry-based)."

    def __init__(self, *, get_raw_func=None, get_epochs_func=None, get_task_func=None, get_subjects_metadata_func=None):
        """Initialize with controller callbacks and bind AI registry to spec."""
        super().__init__(
            registry=ai_registry,
            get_raw_func=get_raw_func,
            get_epochs_func=get_epochs_func,
            get_task_func=get_task_func,
        )
        self.get_subjects_metadata = get_subjects_metadata_func

    def _dataset_from_events(self, task_dto: BaseTaskDTO, params: EpochParamsDTO):
        """Build dataset using event labels."""
        epochs, labels = self.get_epochs(task_dto, params)
        if epochs is None:
            return None, None, {"reason": "epochs_unavailable"}
        epochs.load_data()
        epochs = prepare_channels(epochs, params)
        X = epochs.get_data()
        if labels is None:
            inv_map = {v: k for k, v in (epochs.event_id or {}).items()}
            y = np.array([inv_map.get(code, "?") for code in epochs.events[:, 2]], dtype=object)
        else:
            y = np.array(labels)
        meta = {
            "sfreq": float(epochs.info.get("sfreq", 0.0)),
            "ch_names": list(epochs.ch_names),
            "event_id": dict(epochs.event_id or {}),
            "shape": tuple(X.shape),
        }
        self._log.info("Epoch dataset built (shape=%s) using event labels.", X.shape)
        return X.astype(np.float32), y, meta

    def _dataset_from_participants(self, task_dto: BaseTaskDTO, params: EpochParamsDTO, cols: List[str]):
        """Build dataset using participants metadata columns (cohort preferred).

        For regression targets, continuous numeric columns are returned as a
        float32 matrix (n_samples, n_outputs). For classification-like usage,
        values are combined into pipe-delimited strings. Currently we treat
        participants targets always as regression when columns are numeric.
        """
        def _to_float(v):
            try:
                if v is None:
                    return np.nan
                if isinstance(v, str):
                    s = v.strip().replace(',', '')
                    if s == '' or s.lower() in {"nan", "none", "na", "null"}:
                        return np.nan
                    return float(s)
                return float(v)
            except Exception:
                return np.nan

        if isinstance(task_dto, SubjectFilterDTO):
            task_model = self.get_task(task_dto)
            task_models: List[Any] = getattr(task_model, "task_model_list", [])
            if not task_models:
                return None, None, {"reason": "no_task_models"}
            X_parts: List[np.ndarray] = []
            y_parts: List[Any] = []
            for tm in task_models:
                epochs, _labels = tm.get_epochs(params)
                if epochs is None:
                    continue
                epochs.load_data()
                epochs = prepare_channels(epochs, params)
                subj = getattr(tm.task_dto, "subject", None)
                if subj is None:
                    continue
                X_parts.append(epochs.get_data())
                values = []
                if self.get_subjects_metadata is not None:
                    try:
                        df = self.get_subjects_metadata([subj], cols)
                        if not df.empty:
                            values = [df.iloc[0].get(c) for c in cols]
                    except Exception:  # pragma: no cover
                        values = []
                # Convert to floats when possible (NaN if not convertible)
                if values:
                    vec = [_to_float(v) for v in values]
                    # Numeric regression mode if at least one finite value exists
                    if np.any(np.isfinite(vec)):
                        for _ in range(len(epochs)):
                            y_parts.append(vec)
                    else:
                        # Fall back to classification-like label
                        if len(values) == 1:
                            val = values[0]
                        else:
                            val = "|".join(str(v) for v in values)
                        y_parts.extend([val] * len(epochs))
                else:
                    y_parts.extend([None] * len(epochs))
            if not X_parts:
                return None, None, {"reason": "empty_cohort_epochs"}
            X = np.concatenate(X_parts, axis=0)
            # If first element is list -> regression multi-output
            if y_parts and isinstance(y_parts[0], list):
                y = np.array(y_parts, dtype=np.float32)
                # Drop rows with NaNs in any target
                mask = np.isfinite(y).all(axis=1)
                dropped = int((~mask).sum())
                if dropped > 0:
                    X = X[mask]
                    y = y[mask]
                    self._log.warning("Dropped %d samples with NaN targets (participants cohort).", dropped)
            else:
                y = np.array(y_parts, dtype=object)
            ref_epochs = None
            for tm in task_models:
                e, _ = tm.get_epochs(params)
                if e is not None:
                    ref_epochs = e
                    ref_epochs = prepare_channels(ref_epochs, params)
                    break
            sfreq = float(ref_epochs.info.get("sfreq", 0.0)) if ref_epochs is not None else 0.0
            ch_names = list(ref_epochs.ch_names) if ref_epochs is not None else []
            meta = {"sfreq": sfreq, "ch_names": ch_names, "event_id": {}, "shape": tuple(X.shape), "target_cols": cols}
            self._log.info("Built cohort dataset with participants target %s (shape=%s)", cols, X.shape)
            return X.astype(np.float32), y, meta

        # Single-subject fallback
        epochs, _labels = self.get_epochs(task_dto, params)
        if epochs is None:
            return None, None, {"reason": "epochs_unavailable"}
        epochs.load_data()
        epochs = prepare_channels(epochs, params)
        X = epochs.get_data()
        subj = getattr(task_dto, "subject", None)
        values = []
        if subj and self.get_subjects_metadata is not None:
            try:
                df = self.get_subjects_metadata([subj], cols)
                if not df.empty:
                    values = [df.iloc[0].get(c) for c in cols]
            except Exception:  # pragma: no cover
                values = []
        if values:
            vec = np.array([_to_float(v) for v in values], dtype=np.float32)
            if np.any(np.isfinite(vec)):
                y = np.repeat(vec.reshape(1, -1), repeats=len(X), axis=0).astype(np.float32)
                # If vector contains NaN only, mark unavailable
                if not np.isfinite(vec).all():
                    self._log.warning("Participants target contains NaN; all epochs would inherit NaN labels.")
                    # Drop all rows that are NaN (which is all)
                    return None, None, {"reason": "nan_targets_single_subject", "target_cols": cols}
            else:
                # fall back to classification-like
                if len(values) == 1:
                    val = values[0]
                else:
                    val = "|".join(str(v) for v in values)
                y = np.array([val] * len(X), dtype=object)
        else:
            y = np.array([None] * len(X), dtype=object)
        meta = {
            "sfreq": float(epochs.info.get("sfreq", 0.0)),
            "ch_names": list(epochs.ch_names),
            "event_id": dict(epochs.event_id or {}),
            "shape": tuple(X.shape),
            "warning": "constant_label_single_subject_meta_target",
            "target_cols": cols,
        }
        self._log.info("Single-subject participants target(epochs=%d, target_cols=%s)", len(X), cols)
        return X.astype(np.float32), y, meta

    def _build_epoch_dataset(self, task_dto: BaseTaskDTO, params: EpochParamsDTO) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
        """Return (X, y, meta) using selected target option (event or participants)."""
        sel = getattr(params, "target", ["stimulus"])
        if isinstance(sel, list) and sel:
            sel = sel[0] or "stimulus"
        sel_norm = sel.strip().lower() if isinstance(sel, str) else "stimulus"
        if sel_norm == "stimulus":
            return self._dataset_from_events(task_dto, params)
        if sel_norm == "ccd_accuracy":
            return self._dataset_from_participants(task_dto, params, ["ccd_accuracy"])
        if sel_norm == "ccd_response_time":
            return self._dataset_from_participants(task_dto, params, ["ccd_response_time"])
        if sel_norm.replace(" ", "") in {"ccd_accuracy+ccd_response_time", "ccd_accuracy+ccd_responsetime"}:
            return self._dataset_from_participants(task_dto, params, ["ccd_accuracy", "ccd_response_time"])
        # Fallback
        return self._dataset_from_events(task_dto, params)

