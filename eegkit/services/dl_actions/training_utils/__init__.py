"""Reusable deep-learning training helpers for DL action modules."""

from .data import NumpyDataset, build_dataloaders, split_with_groups
from .evaluation import compute_binary_metrics, eval_loop
from .losses import FocalLoss, build_loss_function, class_weights
from .plotting import build_shaded_error_bar_plot
from .runtime import resolve_device_and_seed
from .training import train_loop

__all__ = [
    "NumpyDataset",
    "split_with_groups",
    "build_dataloaders",
    "train_loop",
    "eval_loop",
    "compute_binary_metrics",
    "FocalLoss",
    "class_weights",
    "build_loss_function",
    "build_shaded_error_bar_plot",
    "resolve_device_and_seed",
]
