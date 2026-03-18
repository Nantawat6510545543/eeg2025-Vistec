"""EEGNet for n-class EEG classification (configurable output head)."""

from __future__ import annotations

import torch
import torch.nn as nn


class Conv2dWithConstraint(nn.Conv2d):
    """Conv2d with per-filter max-norm weight constraint applied during forward."""

    def __init__(self, *args, max_norm: float = 1.0, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class EEGNetNClass(nn.Module):
    """EEGNet for n-class EEG classification from epoch tensors shaped [B, 1, C, T]."""

    DISPLAY_NAME = "EEGNet (n-class classification)"
    DESCRIPTION = (
        "Standard EEGNet architecture for subject-independent n-class EEG classification. "
        "Input shape: [B, 1, C, T]. Output: n class logits."
    )

    def __init__(
        self,
        n_channels: int = 64,
        n_timepoints: int = 128,
        num_classes: int = 4,
        dropout_rate: float = 0.5,
        kern_length: int = 64,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
    ):
        super().__init__()
        if int(num_classes) < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")

        self.num_classes = int(num_classes)

        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern_length), padding=(0, kern_length // 2), bias=False),
            nn.BatchNorm2d(F1),
            Conv2dWithConstraint(F1, F1 * D, (n_channels, 1), groups=F1, bias=False, max_norm=1.0),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(p=dropout_rate),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(p=dropout_rate),
        )

        self.flatten_size = F2 * max(1, n_timepoints // 32)
        self.classifier = nn.Linear(self.flatten_size, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
