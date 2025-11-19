"""Signal-processing utilities: SNR spectrum computation and cleaning helpers."""

from .cleaning_utils import EEGCleaner
from .spectral_utils import snr_spectrum

__all__ = [
    "snr_spectrum",
    "EEGCleaner",
]
