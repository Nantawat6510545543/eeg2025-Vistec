"""AI-related parameter DTOs (training and prediction)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List

from .filter import EpochPSDParamsDTO


@dataclass
class AITrainParamsDTO(EpochPSDParamsDTO):
    """Base training params shared across models."""
    train: List[str] = field(default_factory=lambda: ["epoch", "evoked", "psd"])
    target: List[str] = field(default_factory=lambda: ["stimulus","ccd_accuracy", "ccd_response_time"])
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
class AIPredictParamsDTO(EpochPSDParamsDTO):
    """Prediction params for future checkpoint-based inference."""
    checkpoint_path: Optional[str] = None


@dataclass
class EEGNetMultiRegTrainParamsDTO(AITrainParamsDTO):
    """Training params for EEGNetMultiReg with regression targets.

    Provide selectable regression target options sourced from participants metadata.
    """
    target: List[str] = field(default_factory=lambda: [
        "ccd_accuracy",
        "ccd_response_time",
        "ccd_accuracy + ccd_response_time",
    ])

__all__ = [
    "AITrainParamsDTO",
    "AIPredictParamsDTO",
    "EEGNetMultiRegTrainParamsDTO",
]
