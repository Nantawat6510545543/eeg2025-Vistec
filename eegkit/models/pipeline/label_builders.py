"""Task-specific label builders used by preprocessing and dataset services."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


CCD_TARGET_HIT = "hit"
CCD_TARGET_WRONG = "wrong"
CCD_TARGET_MISS = "miss"


def _norm_token(x) -> str:
    # Normalize heterogeneous event tokens (None/NaN/mixed-case) into a stable lowercase string.
    s = str(x).strip().lower()
    return "" if s in {"", "nan", "none"} else s


def build_ccd_trial_outcomes_and_reaction_times(events_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-trial CCD outcome and reaction time arrays.

    Outcome names:
    - hit: first valid post-target button press matches target side
    - wrong: first valid post-target button press mismatches target side
    - miss: target/press unavailable in the trial window

    Reaction time semantics:
    - RT is measured as first_valid_press_onset - target_onset in seconds for hit trials.
    - Wrong and miss trials are encoded as NaN RT.

    Notes:
    - A press with feedback token "non_target" is ignored.
    - Trial windows are segmented by consecutive contrastTrial_start events.
    """
    if events_df is None or events_df.empty:
        return np.asarray([], dtype=object), np.asarray([], dtype=np.float32)
    if "onset" not in events_df.columns or "value" not in events_df.columns:
        return np.asarray([], dtype=object), np.asarray([], dtype=np.float32)

    # Work on a cleaned, onset-sorted copy so trial windows are deterministic.
    df = events_df.copy()
    df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
    df = df.dropna(subset=["onset"]).sort_values("onset").reset_index(drop=True)

    onsets = df["onset"].to_numpy()
    values = np.asarray([_norm_token(v) for v in df["value"].to_numpy()], dtype=object)
    feedbacks = (
        np.asarray([_norm_token(v) for v in df["feedback"].to_numpy()], dtype=object)
        if "feedback" in df.columns
        else np.asarray(["" for _ in range(len(df))], dtype=object)
    )
    # Trials are segmented by consecutive contrastTrial_start markers.
    trial_starts = np.where(values == "contrasttrial_start")[0]
    if trial_starts.size == 0:
        return np.asarray([], dtype=object), np.asarray([], dtype=np.float32)

    left_target_tokens = {"left_target"}
    right_target_tokens = {"right_target"}
    left_press_tokens = {"left_buttonpress"}
    right_press_tokens = {"right_buttonpress"}

    labels: List[str] = []
    reaction_times: List[float] = []
    for i, s_idx in enumerate(trial_starts):
        t_start = float(onsets[s_idx])
        t_end = float(onsets[trial_starts[i + 1]]) if i + 1 < len(trial_starts) else np.inf
        # Only inspect events strictly inside the current trial window.
        in_win = np.where((onsets > t_start) & (onsets < t_end))[0]

        # Rule 1: find the first target side in the window.
        target_idxs = [idx for idx in in_win if values[idx] in left_target_tokens | right_target_tokens]
        if not target_idxs:
            labels.append(CCD_TARGET_MISS)
            reaction_times.append(float("nan"))
            continue

        target_idx = int(target_idxs[0])
        target_side = "left" if values[target_idx] in left_target_tokens else "right"

        # Rule 2: find first valid press after target (ignore non_target feedback presses).
        press_idxs = [
            idx for idx in in_win
            if idx > target_idx
            and values[idx] in (left_press_tokens | right_press_tokens)
            and feedbacks[idx] != "non_target"
        ]
        if not press_idxs:
            labels.append(CCD_TARGET_MISS)
            reaction_times.append(float("nan"))
            continue

        # Rule 3: side match => hit, side mismatch => wrong.
        press_val = values[int(press_idxs[0])]
        press_side = "left" if press_val in left_press_tokens else "right"
        if press_side == target_side:
            labels.append(CCD_TARGET_HIT)
            rt = float(onsets[int(press_idxs[0])] - onsets[target_idx])
            reaction_times.append(rt if np.isfinite(rt) and rt >= 0.0 else float("nan"))
        else:
            labels.append(CCD_TARGET_WRONG)
            reaction_times.append(float("nan"))

    return np.asarray(labels, dtype=object), np.asarray(reaction_times, dtype=np.float32)


def build_ccd_trial_outcome_labels(events_df: pd.DataFrame) -> np.ndarray:
    """Return one CCD outcome label per trial_start."""
    labels, _reaction_times = build_ccd_trial_outcomes_and_reaction_times(events_df)
    return labels
