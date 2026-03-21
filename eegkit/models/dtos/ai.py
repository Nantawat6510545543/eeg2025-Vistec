"""AI-related parameter DTOs (training and prediction)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# @dataclass
# class AIBaseDTO():
#     train: List[str] = field(default_factory=lambda: ["epoch", "evoked", "psd"])
#     target: List[str] = field(default_factory=lambda: ["stimulus", "accuracy", "ccd_accuracy", "ccd_response_time"])

@dataclass
class DLTrainRuntimeParamsDTO:
    """Runtime/training loop parameters (epochs, batch size, device, seed)."""

    batch_size: int = 32
    epochs_n: int = 100
    lr: float = 0.001
    device: List[str] = field(default_factory=lambda: ["auto", "cpu", "cuda"])
    seed: int = 42
    save_checkpoint: bool = True


@dataclass
class DLTrainSplitParamsDTO:
    """Train/val/test split ratios."""

    val_split: float = 0.2
    test_split: float = 0.2


@dataclass
class DLTrainIOParamsDTO:
    """Dataset/model naming + selection parameters."""

    dataset_path: List[Optional[str]] = field(default_factory=lambda: [None])
    run_name: str = "dl_train"


@dataclass
class DLTrainSchedulerParamsDTO:
    """LR scheduler and early-stopping parameters."""

    scheduler_type: List[str] = field(default_factory=lambda: ["reduce_on_plateau", "none"])
    scheduler_mode: List[str] = field(default_factory=lambda: ["min", "max"])
    scheduler_patience: int = 6
    scheduler_threshold: float = 1e-4
    scheduler_threshold_mode: List[str] = field(default_factory=lambda: ["rel", "abs"])
    scheduler_cooldown: int = 1

    min_lr: float = 1e-6
    lr_factor: float = 0.5
    early_stopping: bool = True
    early_stopping_patience: int = 12


@dataclass
class DLTrainBalanceParamsDTO:
    """Class-imbalance handling strategy."""

    balance_strategy: List[str] = field(
        default_factory=lambda: ["class_weight", "weighted_sampler", "undersample", "none"]
    )


@dataclass
class DLTrainLossParamsDTO:
    """Loss configuration parameters."""

    loss_function: List[str] = field(default_factory=lambda: ["cross_entropy", "focal_loss"])
    label_smoothing: float = 0.0


@dataclass
class DLTrainParamsDTO(
    DLTrainRuntimeParamsDTO,
    DLTrainSplitParamsDTO,
    DLTrainIOParamsDTO,
    DLTrainSchedulerParamsDTO,
    DLTrainBalanceParamsDTO,
    DLTrainLossParamsDTO,
):
    """Base training params shared across deep-learning workflows."""


@dataclass
class EEGNetBinaryTrainParamsDTO(DLTrainParamsDTO):
    """Training params for binary EEGNet."""


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
    "DLTrainRuntimeParamsDTO",
    "DLTrainSplitParamsDTO",
    "DLTrainIOParamsDTO",
    "DLTrainSchedulerParamsDTO",
    "DLTrainBalanceParamsDTO",
    "DLTrainLossParamsDTO",
    "EEGNetBinaryTrainParamsDTO",
    "MLTrainParamsDTO",
    "EEGNetMultiRegTrainParamsDTO",
    "MLTrainDatasetParamsDTO",
    "MLTestDatasetModelParamsDTO",
]
