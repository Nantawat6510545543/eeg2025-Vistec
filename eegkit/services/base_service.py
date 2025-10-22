from __future__ import annotations

from abc import ABC
from typing import Optional, Dict, Any
import numpy as np

# Models are needed for dynamic param preparation
from ..models import EpochParamsDTO


class BaseService(ABC):
    """Abstract base for controller modes (plot / grid_plot / data).

    Responsibilities:
    - Hold human-readable description for UI
    - Build bound spec from a provided registry (name -> {"params", "function"})
    - Provide shared helpers like prepare_params() and _snr_spectrum()

    Subclasses should provide a REGISTRY (dict) or pass one into __init__.
    """

    # Human-readable mode description for the UI
    description: str = ""

    def __init__(
        self,
        *,
        registry: Dict[str, Dict[str, Any]],
        get_raw_func=None,
        get_epochs_func=None,
        get_evoked_func=None,
        get_task_func=None,
    ):
        # Wire controller accessors (subclasses may use a subset)
        self.get_raw = get_raw_func
        self.get_epochs = get_epochs_func
        self.get_evoked = get_evoked_func
        self.get_task = get_task_func

        # Bind registry functions to this instance
        self.spec = registry or {}
        for key in list(self.spec.keys()):
            func = self.spec[key]["function"]
            # Bind the unbound function to this instance
            self.spec[key]["function"] = func.__get__(self)  # type: ignore[attr-defined]

    # ----- Shared helpers -----
    def _snr_spectrum(
        self,
        psd: np.ndarray,
        noise_n_neighbor_freqs: int = 3,
        noise_skip_neighbor_freqs: int = 1,
    ) -> np.ndarray:
        """Compute SNR by dividing PSD by a neighborhood-averaged noise estimate.

        psd shape: (..., n_freqs) -> returns SNR with same shape.
        """
        kernel = np.concatenate(
            (
                np.ones(noise_n_neighbor_freqs),
                np.zeros(2 * noise_skip_neighbor_freqs + 1),
                np.ones(noise_n_neighbor_freqs),
            )
        )
        kernel /= kernel.sum()
        mean_noise = np.apply_along_axis(
            lambda psd_: np.convolve(psd_, kernel, mode="valid"),
            axis=-1,
            arr=psd,
        )
        edge_width = noise_n_neighbor_freqs + noise_skip_neighbor_freqs
        pad = [(0, 0)] * (mean_noise.ndim - 1) + [(edge_width, edge_width)]
        mean_noise = np.pad(mean_noise, pad_width=tuple(pad), constant_values=float("nan"))
        return psd / mean_noise

    def prepare_params(self, task_dto, key):  # optional hook, shared default
        """Return dynamic default values for params widgets.

        Default implementation: if params for the action is an EpochParamsDTO,
        query epochs to discover available labels and pre-fill the 'stimulus'
        dropdown with [None] + labels.
        """
        spec = self.spec[key]
        params_cls = spec.get("params")
        if not params_cls:
            return {}

        try:
            params_obj = params_cls()
        except Exception:
            # If params_cls is not callable, ignore dynamic defaults
            return {}

        if EpochParamsDTO and isinstance(params_obj, EpochParamsDTO):
            if self.get_epochs is None:
                return {}
            _epochs, labels = self.get_epochs(task_dto=task_dto, epoch_params=params_obj)
            if isinstance(labels, str) and labels == "unavailable":
                return {}
            if labels is not None:
                return {"stimulus": [None] + list(labels)}
        return {}
