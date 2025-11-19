"""AI-related parameter DTOs (training and prediction)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List

from .filter import EpochParamsDTO


@dataclass
class AIBaseDTO(EpochParamsDTO):
    """Base AI params including model selection list."""
    model: List[Optional[str]] = field(default_factory=lambda: [None])


@dataclass
class AITrainParamsDTO(AIBaseDTO):
    """Training hyperparameters and target selection for simple trainers."""
    batch_size: int = 32
    epochs_n: int = 1
    lr: float = 0.001
    device: List[str] = field(default_factory=lambda: ["auto", "cpu", "cuda"])
    target: List[str] = field(default_factory=lambda: ["stimulus"])


@dataclass
class AIPredictParamsDTO(AIBaseDTO):
    """Prediction params for future checkpoint-based inference."""
    checkpoint_path: Optional[str] = None

__all__ = [
    "AIBaseDTO",
    "AITrainParamsDTO",
    "AIPredictParamsDTO",
]
