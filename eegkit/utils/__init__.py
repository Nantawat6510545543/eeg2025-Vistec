"""Utilities package

Common helpers for the EEG Workbench, organized into subpackages:
- plot: figure/grid utilities for visualization services
- signal: signal processing routines and cleaning helpers
- channels: channel parsing and selection conveniences
- system: logging and small system helpers
"""

from . import channels as channels  # noqa: F401
# Expose categorized submodules only
from . import plot as plot  # noqa: F401
from . import signal as signal  # noqa: F401
from . import system as system  # noqa: F401

__all__ = [
    "plot",
    "signal",
    "channels",
    "system",
]
