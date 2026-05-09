"""EEGNet for binary EEG classification (2-class output head)."""

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


class EEGNetBinary(nn.Module):
    """EEGNet for binary EEG classification from epoch tensors shaped [B, 1, C, T].

    Architecture follows the reference EEGNet (Lawhern et al. 2018):
    block1 = temporal conv -> depthwise spatial conv -> ELU -> AvgPool -> Dropout
    block2 = separable depthwise conv -> pointwise conv -> ELU -> AvgPool -> Dropout
    classifier = Linear -> 2 class logits (binary).
    """

    DISPLAY_NAME = "EEGNet (binary classification)"
    DESCRIPTION = (
        "Standard EEGNet architecture for subject-independent binary EEG classification. "
        "Input shape: [B, 1, C, T]. Output: 2 class logits."
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
        """Build EEGNet blocks parameterised by electrode count and time dimension."""
        super().__init__()

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

        # Infer classifier input size from the actual conv/pool output shape.
        # This avoids fragile hard-coded formulas that break when padding/pooling
        # yields a different temporal length than expected.
        self.flatten_size = self._infer_flatten_size(n_channels, n_timepoints)
        self.classifier = nn.Linear(int(self.flatten_size), 2)

    def _infer_flatten_size(self, n_channels: int, n_timepoints: int) -> int:
        with torch.no_grad():
            x = torch.zeros(1, 1, int(n_channels), int(n_timepoints), dtype=torch.float32)
            x = self.block1(x)
            x = self.block2(x)
            return int(x.view(1, -1).shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return 2-class logits for input shaped [B, 1, C, T]."""
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
