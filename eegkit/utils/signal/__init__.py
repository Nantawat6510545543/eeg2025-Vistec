"""Signal-processing utilities: SNR spectrum computation and cleaning helpers."""

from .signal_utils import snr_spectrum
from .cleaning_utils import EEGCleaner

__all__ = [
    "snr_spectrum",
    "EEGCleaner",
]
