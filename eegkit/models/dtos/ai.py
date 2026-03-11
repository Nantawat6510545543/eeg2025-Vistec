"""AI-related parameter DTOs (training and prediction)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List


# @dataclass
# class AIBaseDTO():
#     train: List[str] = field(default_factory=lambda: ["epoch", "evoked", "psd"])
#     target: List[str] = field(default_factory=lambda: ["stimulus", "accuracy", "ccd_accuracy", "ccd_response_time"])

@dataclass
class AITrainParamsDTO():
    """Base training params shared across models."""
    batch_size: int = 32
    epochs_n: int = 50
    lr: float = 0.001
    device: List[str] = field(default_factory=lambda: ["auto", "cpu", "cuda"])
    val_split: float = 0.2
    test_split: float = 0.2
    seed: int = 42
    save_checkpoint: bool = True
    weight_classes: bool = False
    patience: int = 0


@dataclass
class AIPredictParamsDTO():
    """Prediction params for future checkpoint-based inference."""
    checkpoint_path: Optional[str] = None


@dataclass
class EEGNetMultiRegTrainParamsDTO():
    """Training params for EEGNetMultiReg with regression targets.

    Provide selectable regression target options sourced from participants metadata.
    """
    target: List[str] = field(default_factory=lambda: [
        "ccd_accuracy",
        "ccd_response_time",
        "ccd_accuracy + ccd_response_time",
    ])


@dataclass
class MLTrainDatasetParamsDTO():
    """Parameters for selecting a discovered dataset and training placeholder settings."""

    dataset_path: List[Optional[str]] = field(default_factory=lambda: [None])
    run_name: str = "ml_train_placeholder"
    estimator: List[str] = field(default_factory=lambda: ["logistic_regression", "random_forest", "svm"])

__all__ = [
    "AITrainParamsDTO",
    "AIPredictParamsDTO",
    "EEGNetMultiRegTrainParamsDTO",
    "MLTrainDatasetParamsDTO",
]
