"""Loss functions and class-weight helpers for DL training."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight


class FocalLoss(nn.Module):
    """Multi-class focal loss for logits."""

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


def class_weights(y, device: torch.device) -> Optional[torch.Tensor]:
    """Compute balanced class weights from labels and move to target device."""
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    labels = np.asarray(y).astype(int)
    classes = np.unique(labels)
    if classes.size <= 1:
        return None
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return torch.from_numpy(np.asarray(cw)).float().to(device)


def build_loss_function(
    name: str,
    class_weight_tensor: Optional[torch.Tensor],
    label_smoothing: float,
) -> nn.Module:
    """Create configured loss module from selection key."""
    if name == "focal_loss":
        return FocalLoss(gamma=2.0, weight=class_weight_tensor)
    return nn.CrossEntropyLoss(weight=class_weight_tensor, label_smoothing=label_smoothing)
