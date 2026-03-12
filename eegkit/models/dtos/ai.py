"""AI-related parameter DTOs (training and prediction)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List


# @dataclass
# class AIBaseDTO():
#     train: List[str] = field(default_factory=lambda: ["epoch", "evoked", "psd"])
#     target: List[str] = field(default_factory=lambda: ["stimulus", "accuracy", "ccd_accuracy", "ccd_response_time"])

@dataclass
class DLTrainParamsDTO():
    """Base training params shared across deep-learning workflows."""
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
class MLTrainParamsDTO():
    """Base training params shared across classical ML workflows."""

    val_split: float = 0.2
    test_split: float = 0.2
    seed: int = 42
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
class MLTrainDatasetParamsDTO(MLTrainParamsDTO):
    """Parameters for selecting dataset and training a classical ML model."""

    dataset_path: List[Optional[str]] = field(default_factory=lambda: [None])
    run_name: str = "ml_train_placeholder"
    estimator: List[str] = field(default_factory=lambda: ["svm", "knn", "random_forest"])
    label_smoothing: float = 0.0


@dataclass
class MLTestDatasetModelParamsDTO(MLTrainParamsDTO):
    """Parameters for testing saved ML models on selected datasets."""

    model_path: List[Optional[str]] = field(default_factory=lambda: [None])
    dataset_path: List[Optional[str]] = field(default_factory=lambda: [None])
    display_mode: List[str] = field(default_factory=lambda: ["true_vs_pred"])

__all__ = [
    "DLTrainParamsDTO",
    "MLTrainParamsDTO",
    "EEGNetMultiRegTrainParamsDTO",
    "MLTrainDatasetParamsDTO",
    "MLTestDatasetModelParamsDTO",
]
