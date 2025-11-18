"""Signal processing helpers category.

Re-exports signal-related utilities so imports can be clearer:

    from eegkit.utils.signal import (snr_spectrum, EEGCleaner)

"""

from .signal_utils import snr_spectrum
from .cleaning_utils import EEGCleaner

__all__ = [
    "snr_spectrum",
    "EEGCleaner",
]
