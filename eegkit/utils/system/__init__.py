"""System and logging helpers category.

    from eegkit.utils.system import (configure_logging, silence_console_logs)

"""

from .logging_utils import configure_logging, silence_console_logs

__all__ = [
    "configure_logging",
    "silence_console_logs",
]
