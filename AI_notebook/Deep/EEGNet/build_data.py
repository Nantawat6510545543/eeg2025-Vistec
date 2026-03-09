"""Build EEGNet training arrays and save them into param-keyed cache."""

import argparse
import importlib.util
import json
import os
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

# Allow importing project modules from workspace root.
sys.path.append(os.path.abspath("/mount/NAS-workspace-portal/eeg2025-Vistec/"))

from eegkit.controller.eeg_controller import EEGController
from eegkit.models.dtos import EpochParamsDTO, SubjectFilterDTO
from eegkit.models.subject_model import EEGSubjectModel
from eegkit.utils.channels import prepare_channels


def _load_cache_utils_module():
    """Load local data_cache_utils module from this folder."""
    module_path = os.path.join(SCRIPT_DIR, "data_cache_utils.py")
    spec = importlib.util.spec_from_file_location("data_cache_utils", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load data_cache_utils from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    """Parse command-line arguments for data building."""
    parser = argparse.ArgumentParser(description="Build and cache X/y/groups/responds for EEGNet.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--task", type=str, default="contrastChangeDetection")
    parser.add_argument("--n_subjects", type=int, default=500)
    parser.add_argument("--channels", type=str, default="69-76,81-83,88,89")
    parser.add_argument("--tmin", type=float, default=-2.0)
    parser.add_argument("--tmax", type=float, default=0.0)
    parser.add_argument("--cache_dir", type=str, default="/mount/NAS-workspace-portal/eeg2025-Vistec/models/data")
    parser.add_argument("--params_json_out", type=str, default="")
    parser.add_argument("--key_prefix", type=str, default="ccd_eegnet")
    return parser.parse_args()


def build_params(args) -> EpochParamsDTO:
    """Create epoch params DTO used by controller.get_epochs."""
    return EpochParamsDTO(
        tmin=args.tmin,
        tmax=args.tmax,
        stimulus=[None],
        channels=args.channels,
        notch=None,
        uv_min=None,
        uv_max=None,
        clean_flatline_sec=None,
        clean_hf_noise_sd_max=None,
        clean_corr_min=None,
        clean_asr_max_std=None,
        clean_power_min_sd=None,
        clean_power_max_sd=None,
        clean_max_outbound_pct=None,
        clean_window_sec=None,
    )


def build_ccd_trial_labels(events_df: pd.DataFrame) -> np.ndarray:
    """Build one binary label per CCD trial start (1=hit, 0=miss)."""
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

    labels = []
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


def extract_epoch_data(subject_model, controller, params, task: str, n_subjects: int):
    """Extract epoch tensors, labels, and subject IDs across selected subjects."""
    group_dto = SubjectFilterDTO(task=task, subject_limit=n_subjects)
    all_task_dtos = subject_model.get_filter_subjects_dto(group_dto)

    epoch_blocks = []
    responds_list = []
    subject_ids = []
    skipped = 0

    for dto in all_task_dtos:
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
        raise RuntimeError("No valid epoch blocks were collected.")

    X_blocks = [ep.get_data().astype(np.float32, copy=False) for ep in epoch_blocks]
    X_ct = np.concatenate(X_blocks, axis=0)
    if X_ct.ndim != 3:
        raise AssertionError(f"Expected concatenated epoch data shape (S*N, C, T), got {X_ct.shape}")
    X = np.expand_dims(X_ct, axis=1)
    if X.ndim != 4:
        raise AssertionError(f"Expected EEGNet input shape (S*N, D, C, T), got {X.shape}")
    if X.shape[1] != 1:
        raise AssertionError(f"Expected depth axis D=1, got D={X.shape[1]}")

    responds = np.asarray(responds_list, dtype=np.int64)
    y = responds.copy()
    groups = np.asarray(subject_ids).astype(str)

    if X.shape[0] != len(y):
        raise AssertionError(f"Mismatch: X rows ({X.shape[0]}) != y length ({len(y)})")
    if len(y) != len(groups):
        raise AssertionError(f"Mismatch: y length ({len(y)}) != groups length ({len(groups)})")

    print(f"Runs considered: {len(all_task_dtos)}")
    print(f"Runs skipped: {skipped}")
    print(f"X shape (S*N,D,C,T): {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"groups shape: {groups.shape}")

    return X, y, groups, responds


def params_to_dict(params_obj):
    """Convert params object to dictionary for JSON export."""
    if is_dataclass(params_obj):
        return asdict(params_obj)
    if hasattr(params_obj, "__dict__"):
        return dict(params_obj.__dict__)
    return {"params_repr": str(params_obj)}


def main():
    """Build arrays and cache them using params-derived key."""
    args = parse_args()

    cache_utils = _load_cache_utils_module()
    build_cache_key = cache_utils.build_cache_key
    save_cache = cache_utils.save_cache

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    subject_model = EEGSubjectModel(data_dir)
    controller = EEGController(subject_model)

    params = build_params(args)
    X, y, groups, responds = extract_epoch_data(
        subject_model=subject_model,
        controller=controller,
        params=params,
        task=args.task,
        n_subjects=args.n_subjects,
    )

    cache_key = build_cache_key(params_obj=params, n_subjects=args.n_subjects, key_prefix=args.key_prefix)
    paths = save_cache(
        cache_dir=Path(args.cache_dir),
        cache_key=cache_key,
        X=X,
        y=y,
        groups=groups,
        responds=responds,
        extra_meta={"task": args.task, "n_subjects": args.n_subjects},
    )

    print("Saved cache files:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    print(f"cache_key: {cache_key}")

    if args.params_json_out:
        out_path = Path(args.params_json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(params_to_dict(params), indent=2, sort_keys=True), encoding="utf-8")
        print(f"params_json_out: {out_path}")


if __name__ == "__main__":
    main()
