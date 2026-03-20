"""Data splitting and DataLoader helpers for DL training."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from torch.utils.data import DataLoader, Dataset as TorchDataset, WeightedRandomSampler

import logging
logger = logging.getLogger(__name__)
class NumpyDataset(TorchDataset):
    """Wrap numpy arrays as a minimal torch Dataset."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.as_tensor(self.x[idx], dtype=torch.float32),
            torch.as_tensor(self.y[idx], dtype=torch.long),
        )


def split_with_groups(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    test_split: float,
    val_split: float,
    seed: int,
):
    """Subject-independent train/val/test split with configurable ratios."""
    if test_split <= 0.0 or test_split >= 1.0:
        raise ValueError(f"test_split must be in (0, 1), got {test_split}")
    if val_split <= 0.0 or val_split >= 1.0:
        raise ValueError(f"val_split must be in (0, 1), got {val_split}")
    if test_split + val_split >= 1.0:
        raise ValueError(
            f"test_split + val_split must be < 1.0, got {test_split + val_split}"
        )

    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_split, random_state=seed)
    trainval_idx, test_idx = next(gss1.split(x, y, groups=groups))

    x_tv, y_tv, g_tv = x[trainval_idx], y[trainval_idx], groups[trainval_idx]
    val_ratio = val_split / (1.0 - test_split)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed + 1)
    train_idx, val_idx = next(gss2.split(x_tv, y_tv, groups=g_tv))

    return (
        x_tv[train_idx], y_tv[train_idx],
        x_tv[val_idx], y_tv[val_idx],
        x[test_idx], y[test_idx],
    )


def split_with_groups_auto(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    test_split: float = 0.2,
    val_split: float = 0.2,
    seed: int = 42,
    n_subjects_threshold: int = 5,
):
    """Choose between GroupShuffleSplit and LeaveOneGroupOut based on subject count.
    
    If number of unique subjects < threshold, uses LeaveOneGroupOut CV (leave one subject out as test).
    Otherwise, uses standard GroupShuffleSplit with fixed train/val/test ratios.
    
    Args:
        x: Input features (N, n_features, ...)
        y: Labels (N,)
        groups: Subject/group assignments (N,)
        test_split: Test set ratio if using GroupShuffleSplit (default 0.2 = 20%)
        val_split: Validation set ratio if using GroupShuffleSplit (default 0.2 = 20%)
        seed: Random seed
        n_subjects_threshold: If n_unique_subjects < this, use LeaveOneGroupOut (default 5)
    
    Returns:
        Tuple of (x_train, y_train, x_val, y_val, x_test, y_test)
    """
    n_unique_subjects = len(np.unique(groups))
    
    if n_unique_subjects < n_subjects_threshold:
        logger.info(
            f"Detected {n_unique_subjects} subjects (< {n_subjects_threshold}). "
            "Using LeaveOneGroupOut CV for better small-cohort generalization."
        )
        return _split_with_logo(x, y, groups, val_split=val_split, seed=seed)
    else:
        logger.info(
            f"Detected {n_unique_subjects} subjects (>= {n_subjects_threshold}). "
            "Using standard GroupShuffleSplit."
        )
        return split_with_groups(x, y, groups, test_split=test_split, val_split=val_split, seed=seed)


def _split_with_logo(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    val_split: float,
    seed: int,
):
    """Leave-One-Group-Out (LOGO) for small cohorts: one group as test, split remainder into train/val."""
    logo = LeaveOneGroupOut()
    splits = list(logo.split(x, groups=groups))
    
    if len(splits) == 0:
        raise ValueError("Cannot perform LOGO split with current data")
    
    # Use first split: one group left out as test, remaining as train+val
    trainval_idx, test_idx = splits[0]
    
    x_tv, y_tv, g_tv = x[trainval_idx], y[trainval_idx], groups[trainval_idx]
    x_te, y_te = x[test_idx], y[test_idx]
    
    # Further split trainval into train/val using GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
    train_idx, val_idx = next(gss.split(x_tv, y_tv, groups=g_tv))
    
    logger.info(
        f"LOGO split: test_group={np.unique(groups[test_idx])}, "
        f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
    )
    
    return (
        x_tv[train_idx], y_tv[train_idx],
        x_tv[val_idx], y_tv[val_idx],
        x_te, y_te,
    )



def build_dataloaders(
    x_tr,
    y_tr,
    x_val,
    y_val,
    x_te,
    y_te,
    batch_size: int,
    weighted_sampler: bool,
    undersample: bool = False,
    seed: int = 42,
):
    """Return train / val / test DataLoaders."""
    if undersample:
        before_counts = dict(zip(*np.unique(y_tr, return_counts=True)))
        classes, counts = np.unique(y_tr, return_counts=True)
        if classes.size >= 2 and counts.min() > 0:
            target_n = int(counts.min())
            rng = np.random.default_rng(seed)
            sampled_idx = []
            for c in classes:
                cls_idx = np.where(y_tr == c)[0]
                keep_idx = rng.choice(cls_idx, size=target_n, replace=False)
                sampled_idx.append(keep_idx)
            sampled_idx = np.concatenate(sampled_idx)
            rng.shuffle(sampled_idx)
            x_tr = x_tr[sampled_idx]
            y_tr = y_tr[sampled_idx]
            after_counts = dict(zip(*np.unique(y_tr, return_counts=True)))
            logger.info(
                "Applied undersampling: train class counts before=%s after=%s",
                before_counts,
                after_counts,
            )
        else:
            logger.warning("Undersampling requested but train split is not suitable; skipping undersample.")

    sampler: Optional[WeightedRandomSampler] = None
    shuffle = True
    if weighted_sampler:
        counts = np.bincount(y_tr, minlength=2)
        counts = np.maximum(counts, 1)
        w = (1.0 / counts)[y_tr]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(w, dtype=torch.double),
            num_samples=len(w),
            replacement=True,
        )
        shuffle = False

    tr_dl = DataLoader(
        NumpyDataset(x_tr, y_tr),
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=False,
    )
    val_dl = DataLoader(NumpyDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    te_dl = DataLoader(NumpyDataset(x_te, y_te), batch_size=batch_size, shuffle=False)

    logger.info("DataLoader Shapes: train=%s val=%s test=%s", tr_dl.dataset.x.shape, val_dl.dataset.x.shape, te_dl.dataset.x.shape)
    return tr_dl, val_dl, te_dl
