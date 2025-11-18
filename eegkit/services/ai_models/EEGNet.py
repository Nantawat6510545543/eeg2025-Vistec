"""Compact CNN example (EEGNet-inspired) with single binary output head for demos."""

import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    """Small convolutional network baseline for binary EEG classifications."""

    DISPLAY_NAME = "EEGNet (binary head)"
    DESCRIPTION = (
        "Compact CNN for EEG-like 2D inputs; example architecture with a single FC head "
        "suited for binary/continuous outputs in demos."
    )

    def __init__(self):
        """Initialize convolutional layers and single-head classifier."""
        super(EEGNet, self).__init__()
        self.T = 120

        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, 64), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)

        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)

        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))

        # FC Layer
        self.fc1 = nn.Linear(4 * 2 * 7, 1)

    def forward(self, x):
        """Compute prediction for input tensor shaped [B, 1, C, T]."""
        # Layer 1
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        x = x.permute(0, 3, 1, 2)

        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)

        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)

        # FC Layer
        x = x.view(-1, 4 * 2 * 7)
        x = F.sigmoid(self.fc1(x))
        return x
