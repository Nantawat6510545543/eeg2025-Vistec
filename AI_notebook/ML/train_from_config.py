"""Build EEG training data and train ML model from a JSON config."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from tqdm.auto import tqdm

import sys
sys.path.append(os.path.abspath("/mount/NAS-workspace-portal/eeg2025-Vistec/"))
from eegkit.controller.eeg_controller import EEGController
from eegkit.models.dtos import EpochParamsDTO, SubjectFilterDTO
from eegkit.models.subject_model import EEGSubjectModel
from eegkit.utils.channels import prepare_channels
from eegkit.utils.system.logging_utils import configure_logging, silence_console_logs

sys.path.append(os.path.abspath("/mount/NAS-workspace-portal/eeg2025-Vistec/AI_notebook/ML"))
from data_cache_utils import load_or_cache
from extration_EEG import extract_nonlinear_features, extract_time_freq_features
from model_factory import KNN, RandomForest, SVM


log = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict[str, Any]:
    """Load and return JSON configuration."""
    log.info("Loading config from %s", config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    log.info("Config loaded. Sections: %s", ", ".join(sorted(cfg.keys())))
    return cfg


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


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

    labels: list[int] = []
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


def build_epoch_dataset(controller: EEGController, subject_model: EEGSubjectModel, cfg: dict[str, Any]):
    """Build epoch tensor and aligned labels/groups using config options."""
    data_cfg = cfg["data"]
    task = data_cfg.get("task", "contrastChangeDetection")
    n_subjects = int(data_cfg.get("n_subjects", 100))

    epoch_cfg = cfg["epoch_params"]
    params = EpochParamsDTO(
        tmin=float(epoch_cfg.get("tmin", -2.0)),
        tmax=float(epoch_cfg.get("tmax", 0.0)),
        stimulus=epoch_cfg.get("stimulus", [None]),
        channels=str(epoch_cfg.get("channels", "69-76,81-83,88,89")),
        notch=epoch_cfg.get("notch", None),
        uv_min=epoch_cfg.get("uv_min", None),
        uv_max=epoch_cfg.get("uv_max", None),
        clean_flatline_sec=epoch_cfg.get("clean_flatline_sec", None),
        clean_hf_noise_sd_max=epoch_cfg.get("clean_hf_noise_sd_max", None),
        clean_corr_min=epoch_cfg.get("clean_corr_min", None),
        clean_asr_max_std=epoch_cfg.get("clean_asr_max_std", None),
        clean_power_min_sd=epoch_cfg.get("clean_power_min_sd", None),
        clean_power_max_sd=epoch_cfg.get("clean_power_max_sd", None),
        clean_max_outbound_pct=epoch_cfg.get("clean_max_outbound_pct", None),
        clean_window_sec=epoch_cfg.get("clean_window_sec", None),
    )

    group_dto = SubjectFilterDTO(task=task, subject_limit=n_subjects)
    all_task_dtos = subject_model.get_filter_subjects_dto(group_dto)
    log.info(
        "Building epochs for task=%s, requested_subject_limit=%d, discovered_runs=%d",
        task,
        n_subjects,
        len(all_task_dtos),
    )

    epoch_blocks = []
    responds_list: list[int] = []
    subject_ids: list[str] = []

    skipped = 0
    t0 = time.perf_counter()
    for dto in tqdm(all_task_dtos, desc="Collecting epochs", unit="run"):
        task_model = subject_model.get_task(dto)
        events_df = task_model.get_event()
        trial_labels = build_ccd_trial_labels(events_df)
        if trial_labels.size == 0:
            skipped += 1
            continue

        epoch, _ = controller.get_epochs(dto, params)
        if epoch is None:
            skipped += 1
            continue

        epoch = prepare_channels(epoch, params)

        selection = np.asarray(getattr(epoch, "selection", np.arange(len(epoch), dtype=int)), dtype=int)
        valid_pos = np.where((selection >= 0) & (selection < len(trial_labels)))[0]
        if valid_pos.size == 0:
            skipped += 1
            continue

        epoch_kept = epoch[valid_pos]
        labels_kept = trial_labels[selection[valid_pos]]

        epoch_blocks.append(epoch_kept)
        responds_list.extend(labels_kept.tolist())
        subject_ids.extend([dto.subject] * len(epoch_kept))

    if not epoch_blocks:
        raise RuntimeError("No valid epochs were collected. Check task/filter/config.")

    x_blocks = [ep.get_data().astype(np.float32, copy=False) for ep in epoch_blocks]
    x_ct = np.concatenate(x_blocks, axis=0)  # (N, C, T)
    y = np.asarray(responds_list, dtype=np.int64)
    groups = np.asarray(subject_ids)

    if x_ct.shape[0] != len(y):
        raise AssertionError(f"Mismatch: X rows ({x_ct.shape[0]}) != y length ({len(y)})")
    if len(y) != len(groups):
        raise AssertionError(f"Mismatch: y length ({len(y)}) != groups length ({len(groups)})")

    print(f"Runs considered: {len(all_task_dtos)}")
    print(f"Runs skipped: {skipped}")
    print(f"Collected epochs: {x_ct.shape[0]}")
    log.info(
        "Epoch build done: total_epochs=%d, skipped_runs=%d, elapsed=%.2fs",
        x_ct.shape[0],
        skipped,
        time.perf_counter() - t0,
    )
    return x_ct, y, groups, params


def to_feature_matrix(x_ct: np.ndarray, fs: float, feature_mode: str) -> np.ndarray:
    """Convert (N, C, T) epochs into (N, F) features."""
    mode = feature_mode.lower().strip()
    if mode == "flatten":
        log.info("Feature mode=flatten. Reshaping (N,C,T) -> (N,F).")
        return x_ct.reshape(x_ct.shape[0], -1)

    if mode != "paper18":
        raise ValueError("features.mode must be 'flatten' or 'paper18'.")

    log.info("Feature mode=paper18. Extracting time/freq + nonlinear features per epoch/channel.")
    feature_rows = []
    for epoch in tqdm(x_ct, desc="Extracting features", unit="epoch"):
        ch_feats = []
        for signal in epoch:
            time_freq = extract_time_freq_features(signal, fs)
            nonlinear = extract_nonlinear_features(signal)
            combined = {**time_freq, **nonlinear}
            ch_feats.extend(combined.values())
        feature_rows.append(ch_feats)

    return np.asarray(feature_rows, dtype=np.float32)


def split_subject_independent(x: np.ndarray, y: np.ndarray, groups: np.ndarray, test_size: float, val_size: float, seed: int):
    """Split into train/val/test with subject-independent groups."""
    log.info(
        "Splitting dataset with GroupShuffleSplit: test_size=%.3f, val_size=%.3f, seed=%d",
        test_size,
        val_size,
        seed,
    )
    gss_outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(gss_outer.split(x, y, groups=groups))

    x_train_val, y_train_val, g_train_val = x[train_val_idx], y[train_val_idx], groups[train_val_idx]

    relative_val = val_size / max(1e-9, (1.0 - test_size))
    gss_inner = GroupShuffleSplit(n_splits=1, test_size=relative_val, random_state=seed + 1)
    train_idx_rel, val_idx_rel = next(gss_inner.split(x_train_val, y_train_val, groups=g_train_val))

    x_train, y_train = x_train_val[train_idx_rel], y_train_val[train_idx_rel]
    x_val, y_val = x_train_val[val_idx_rel], y_train_val[val_idx_rel]
    x_test, y_test = x[test_idx], y[test_idx]

    log.info(
        "Split done: train=%d, val=%d, test=%d",
        len(x_train),
        len(x_val),
        len(x_test),
    )

    return x_train, y_train, x_val, y_val, x_test, y_test


def get_model(model_type: str, model_dir: Path, model_name: str):
    """Create model instance from model type."""
    model_map = {
        "svm": SVM,
        "knn": KNN,
        "random_forest": RandomForest,
    }
    key = model_type.lower().strip()
    if key not in model_map:
        raise ValueError("model.type must be one of: svm, knn, random_forest")

    log_path = str(model_dir)
    if not log_path.endswith(os.sep):
        log_path += os.sep
    log.info("Initializing model: type=%s, name=%s, model_dir=%s", key, model_name, log_path)
    return model_map[key](log_path=log_path, model_name=model_name)


def main():
    """Run config-driven data building, training, and evaluation."""
    parser = argparse.ArgumentParser(description="Config-driven EEG ML training pipeline")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    print(f"[PIPELINE] Config: {config_path}")

    configure_logging(cfg.get("logging", {}).get("level", "INFO"))
    silence_console_logs(cfg.get("logging", {}).get("mne_level", "WARNING"))
    log.info("Starting config-driven training pipeline")

    data_dir = Path(cfg["data"]["data_dir"]).resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    print(f"[PIPELINE] Data dir: {data_dir}")
    log.info("Data directory: %s", data_dir)

    output_cfg = cfg["output"]
    model_dir = ensure_dir(output_cfg.get("model_dir", "./models"))
    report_dir = ensure_dir(output_cfg.get("report_dir", "./reports"))

    subject_model = EEGSubjectModel(data_dir)
    controller = EEGController(subject_model)
    print("[PIPELINE] Subject model and controller initialized")

    feature_cfg = cfg.get("features", {})
    fs = float(feature_cfg.get("fs", cfg.get("epoch_params", {}).get("resample_fs", 100.0)))
    feature_mode = feature_cfg.get("mode", "flatten")

    cache_cfg = cfg.get("cache", {})
    cache_enabled = bool(cache_cfg.get("enabled", True))
    log.info("Dataset cache enabled=%s", cache_enabled)
    print(f"[PIPELINE] Dataset cache enabled: {cache_enabled}")

    if cache_enabled:
        cache_dir = Path(cache_cfg.get("dir", "./dataset_cache")).resolve()
        key_prefix = str(cache_cfg.get("key_prefix", "ml_config"))
        n_subjects = int(cfg["data"].get("n_subjects", 100))
        print(f"[PIPELINE] Cache dir: {cache_dir}")
        print(f"[PIPELINE] Cache key prefix: {key_prefix}")

        cache_params_obj = {
            "data": cfg.get("data", {}),
            "epoch_params": cfg.get("epoch_params", {}),
            "features": {
                "mode": feature_mode,
                "fs": fs,
            },
        }

        def _build_fn():
            x_ct_local, y_local, groups_local, _params = build_epoch_dataset(controller, subject_model, cfg)
            x_local = to_feature_matrix(x_ct_local, fs=fs, feature_mode=feature_mode)
            return x_local, y_local, groups_local, y_local

        cached = load_or_cache(
            cache_dir=cache_dir,
            params_obj=cache_params_obj,
            n_subjects=n_subjects,
            build_fn=_build_fn,
            key_prefix=key_prefix,
            require_responds=False,
        )

        x = np.asarray(cached["X"], dtype=np.float32)
        y = np.asarray(cached["y"], dtype=np.int64)
        groups = np.asarray(cached["groups"]).astype(str)
        cache_status = cached.get("status", "unknown")
        cache_key = cached.get("cache_key", None)
        print(f"Dataset cache status: {cache_status}")
        if cache_key is not None:
            print(f"Dataset cache key: {cache_key}")
        log.info("Dataset cache status=%s, key=%s", cache_status, cache_key)
    else:
        x_ct, y, groups, _params = build_epoch_dataset(controller, subject_model, cfg)
        x = to_feature_matrix(x_ct, fs=fs, feature_mode=feature_mode)
        y = np.asarray(y, dtype=np.int64)
        groups = np.asarray(groups).astype(str)
        cache_status = "disabled"
        cache_key = None
        log.info("Cache disabled; dataset built from source.")

    split_cfg = cfg.get("split", {})
    print("[PIPELINE] Splitting dataset (subject-independent)")
    x_train, y_train, x_val, y_val, x_test, y_test = split_subject_independent(
        x,
        y,
        groups,
        test_size=float(split_cfg.get("test_size", 0.2)),
        val_size=float(split_cfg.get("val_size", 0.2)),
        seed=int(split_cfg.get("seed", 42)),
    )

    model_cfg = cfg["model"]
    print(f"[PIPELINE] Model: {model_cfg.get('type', 'svm')} ({model_cfg.get('name', 'baseline')})")
    model = get_model(
        model_type=model_cfg.get("type", "svm"),
        model_dir=model_dir,
        model_name=model_cfg.get("name", "baseline"),
    )

    print("[PIPELINE] Training started")
    model.fit(x_train, y_train, x_val, y_val)
    print("[PIPELINE] Training finished, running evaluation")
    y_dict, evaluation = model.predict(x_test, y_test)
    log.info("Training + evaluation complete. Metrics: %s", evaluation)

    run_summary = {
        "config_path": str(config_path),
        "data_dir": str(data_dir),
        "feature_mode": feature_mode,
        "x_shape": list(x.shape),
        "label_distribution": {str(int(k)): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "n_subjects": int(len(np.unique(groups))),
        "cache": {
            "enabled": cache_enabled,
            "status": cache_status,
            "key": cache_key,
        },
        "split_shapes": {
            "train": [int(x_train.shape[0]), int(x_train.shape[1])],
            "val": [int(x_val.shape[0]), int(x_val.shape[1])],
            "test": [int(x_test.shape[0]), int(x_test.shape[1])],
        },
        "model": {
            "type": model_cfg.get("type", "svm"),
            "name": model_cfg.get("name", "baseline"),
            "model_dir": str(model_dir),
        },
        "evaluation": evaluation,
    }

    report_path = report_dir / f"{model_cfg.get('name', 'baseline')}_summary.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    print(f"Training complete. Summary saved to: {report_path}")
    print(f"Evaluation: {evaluation}")
    log.info("Summary written to %s", report_path)


if __name__ == "__main__":
    main()
