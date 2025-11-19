"""EEGNet variant with three classification heads (background/foreground/stimulus)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNetMultiOutput(nn.Module):
    """Multi-head CNN producing logits for separate background, foreground and stimulus tasks."""

    DISPLAY_NAME = "EEGNet (multi-output)"
    DESCRIPTION = (
        "EEGNet variant with three classification heads (background/foreground/stimulus); "
        "useful for multi-task setups."
    )

    def __init__(self, n_classes=(2, 4, 3)):
        """Initialize shared CNN trunk and three task-specific heads."""
        super().__init__()
        self.n_classes = n_classes

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

        # LazyLinear heads (infer in_features automatically)
        self.fc_bg = nn.LazyLinear(self.n_classes[0])
        self.fc_fg = nn.LazyLinear(self.n_classes[1])
        self.fc_stim = nn.LazyLinear(self.n_classes[2])

    def forward(self, x):
        """Return tuple of logits for (background, foreground, stimulus)."""
        if x.ndim == 3:
            x = x.unsqueeze(1)

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

        out_bg = self.fc_bg(x)
        out_fg = self.fc_fg(x)
        out_stim = self.fc_stim(x)
        return out_bg, out_fg, out_stim
