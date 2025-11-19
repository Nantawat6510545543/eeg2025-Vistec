"""Stateless helper utilities for AI actions (model discovery, adapters, loaders)."""

from __future__ import annotations

import logging
from importlib import import_module
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
except Exception:  # pragma: no cover - optional dependency in non-AI flows
    torch = None  # type: ignore
    nn = None  # type: ignore
    TensorDataset = object  # type: ignore
    DataLoader = object  # type: ignore


logger = logging.getLogger(__name__)


def model_metadata() -> Dict[str, Dict[str, Any]]:
    """Return mapping of model_name -> {display_name, description} discovered under ai_models."""
    meta: Dict[str, Dict[str, Any]] = {}
    try:
        mod = import_module('eegkit.services.ai_models')
        for attr in getattr(mod, '__all__', []):
            obj = getattr(mod, attr, None)
            if nn and isinstance(obj, type) and issubclass(obj, nn.Module):
                display = getattr(obj, 'DISPLAY_NAME', attr)
                desc = getattr(obj, 'DESCRIPTION', (obj.__doc__ or '').strip())
                meta[attr] = {
                    'display_name': display,
                    'description': desc,
                }
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.debug("Failed to inspect ai_models: %s", exc)
    if not meta:
        meta['none'] = {
            'display_name': 'No models found',
            'description': 'No models discovered under eegkit.services.ai_models. '
                           'Add model classes and export them via __all__.',
        }
    return meta


def available_models() -> List[str]:
    """Return available model names sorted for stable UI lists."""
    return sorted(model_metadata().keys())


def select_device(pref: Optional[str] = None):
    """Return a torch.device respecting preference with CUDA auto-fallback."""
    if torch is None:
        return None
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode_labels(y: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Encode arbitrary labels as indices and return (indices, class_names)."""
    uniq = np.unique(y)
    index = {v: i for i, v in enumerate(uniq)}
    y_idx = np.array([index[v] for v in y], dtype=np.int64)
    classes = [str(v) for v in uniq.tolist()]
    return y_idx, classes


def make_dataloader(X: np.ndarray, y_idx: np.ndarray, batch_size: int):
    """Build a PyTorch DataLoader from numpy arrays."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch not available for dataloader creation")
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y_idx)
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=int(batch_size), shuffle=True)
