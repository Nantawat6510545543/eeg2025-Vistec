"""Data splitting and DataLoader helpers for DL training."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
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


def build_dataloaders(
    x_tr, y_tr, x_val, y_val, x_te, y_te, batch_size: int, weighted_sampler: bool
):
    """Return train / val / test DataLoaders."""
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
