"""Runtime helpers for device selection and seed setup."""

from __future__ import annotations

import random

import numpy as np
import torch


def resolve_device_and_seed(device_value, seed: int):
    """Resolve device selector and apply deterministic seeds."""
    device_str = device_value or "cpu"
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    elif str(device_str).startswith("cuda") and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return str(device_str), device
