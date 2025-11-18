import torch.nn as nn


class SimpleNN(nn.Module):
    DISPLAY_NAME = "SimpleNN (MLP)"
    DESCRIPTION = (
        "Minimal fully-connected network; expects flattened features; "
        "good for smoke tests and tabular-like baselines."
    )

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.layers(x)
