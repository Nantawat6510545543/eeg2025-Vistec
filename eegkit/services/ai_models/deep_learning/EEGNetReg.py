"""Single-target EEGNet regressor for regression prediction."""

from __future__ import annotations

import torch
import torch.nn as nn

from .EEGNetBinary import EEGNetBinary


class EEGNetReg(EEGNetBinary):
    """EEGNet regression model specialized for one continuous target."""

    DISPLAY_NAME = "EEGNet (reg)"
    DESCRIPTION = (
        "EEGNetBinary-based regressor with the same feature extractor blocks and "
        "a single-unit regression head for continuous targets."
    )

    def __init__(
        self,
        n_channels: int = 64,
        n_timepoints: int = 128,
        dropout_rate: float = 0.5,
        kern_length: int = 64,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
    ):
        super().__init__(
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            dropout_rate=dropout_rate,
            kern_length=kern_length,
            F1=F1,
            D=D,
            F2=F2,
        )
        self.regressor = nn.Linear(self.flatten_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return predictions with shape (batch,)."""
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        out = self.regressor(x)
        return out.squeeze(-1)
