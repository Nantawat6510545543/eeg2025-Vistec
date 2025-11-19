"""EEGNet trunk with configurable multi-output regression head(s)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNetMultiReg(nn.Module):
    """EEGNet variant producing continuous outputs for regression tasks."""

    DISPLAY_NAME = "EEGNet (multi-regression)"
    DESCRIPTION = (
        "EEGNet variant with a shared convolutional trunk and one LazyLinear head "
        "emitting N continuous regression targets (e.g., ccd_accuracy, ccd_response_time)."
    )

    def __init__(self, n_outputs: int = 1):
        """Initialize shared CNN trunk and a single regression head."""
        super().__init__()
        self.n_outputs = int(n_outputs)

        self.conv1 = nn.Conv2d(1, 16, (1, 64))
        self.batchnorm1 = nn.BatchNorm2d(16, affine=False)

        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(16, 32, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(32, affine=False)
        self.pooling2 = nn.MaxPool2d((2, 4))

        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(32, 64, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(64, affine=False)
        self.pooling3 = nn.MaxPool2d((2, 4))

        # Regression head (LazyLinear infers in_features after first forward pass)
        self.fc_reg = nn.LazyLinear(self.n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return continuous predictions with shape (batch_size, n_outputs)."""
        if x.ndim == 3:  # (batch, channels, time)
            x = x.unsqueeze(1)  # add spatial dim for conv2d trunk

        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)

        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)

        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)

        x = torch.flatten(x, start_dim=1)
        return self.fc_reg(x)
