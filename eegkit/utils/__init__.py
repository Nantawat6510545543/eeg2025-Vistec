"""Utilities package

API: import from categorized subpackages for clarity.

- utils.plot: plotting/figure helpers
- utils.signal: signal processing and cleaning
- utils.channels: channel parsing/selection
- utils.system: logging and system helpers

Examples:
    from eegkit.utils.plot import render_label_grid, finalize_figure
    from eegkit.utils.signal import snr_spectrum, EEGCleaner
    from eegkit.utils.system import configure_logging
"""

# Expose categorized submodules only
from . import plot as plot  # noqa: F401
from . import signal as signal  # noqa: F401
from . import channels as channels  # noqa: F401
from . import system as system  # noqa: F401

__all__ = [
    "plot",
    "signal",
    "channels",
    "system",
]
