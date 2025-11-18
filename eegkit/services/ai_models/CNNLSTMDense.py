"""Temporal model combining Conv1d feature extractor, LSTM and dense classifier."""

import torch.nn as nn


class CNNLSTMDense(nn.Module):
    """Conv1d + LSTM architecture over [B, C, T] input producing class logits."""

    DISPLAY_NAME = "CNN + LSTM + Dense"
    DESCRIPTION = (
        "Temporal model: Conv1d feature extractor over channels followed by LSTM and dense classifier; "
        "input shape [B, C, T]."
    )
    """
    Input:  x of shape [B, C_in, T]
    Blocks: Conv1d -> Conv1d -> (transpose) -> LSTM -> Dense classifier
    """

    def __init__(
            self,
            in_channels: int,  # e.g., EEG channels
            num_classes: int,
            lstm_hidden: int = 128,
            lstm_layers: int = 1,
            bidirectional: bool = True,
            dropout: float = 0.5,
            pool: str = "mean",  # "mean" or "last" over the LSTM outputs
    ):
        """Initialize CNN feature extractor, LSTM, and classifier head."""
        super().__init__()
        # --- CNN feature extractor ---
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # after CNN we’ll have [B, 128, T’]
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(128, num_classes),
        )

        assert pool in ("mean", "last")
        self.pool = pool

    def forward(self, x):
        """Compute logits from input tensor shaped [B, C_in, T]."""
        # x: [B, C_in, T]
        x = self.cnn(x)  # [B, 128, T’]
        x = x.transpose(1, 2)  # [B, T’, 128] for LSTM (batch_first=True)
        y, _ = self.lstm(x)  # [B, T’, H]

        if self.pool == "mean":
            y = y.mean(dim=1)  # temporal mean-pool
        else:
            y = y[:, -1, :]  # last timestep

        logits = self.classifier(y)  # [B, num_classes]
        return logits
