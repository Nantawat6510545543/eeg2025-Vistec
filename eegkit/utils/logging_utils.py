from __future__ import annotations

import logging
import os
from typing import Optional


def configure_logging(level: Optional[str] = None) -> None:
    if logging.getLogger().handlers:
        return

    level_name = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    try:
        lvl = getattr(logging, level_name)
    except Exception:
        lvl = logging.INFO

    logging.basicConfig(
        level=lvl,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def silence_console_logs(min_level: Optional[str] = "WARNING") -> None:
    """
    Reduce verbosity of console (stream) logging so UI/notebook output stays clean.
    Does not touch file handlers or other handlers; only raises the level of StreamHandlers.
    Example (in notebook):
        from eegkit.utils.logging_utils import silence_console_logs
        silence_console_logs("WARNING")
    """
    try:
        level_name = (min_level or "WARNING").upper()
        lvl = getattr(logging, level_name, logging.WARNING)
    except Exception:
        lvl = logging.WARNING
    root = logging.getLogger()
    for h in list(root.handlers):
        try:
            if isinstance(h, logging.StreamHandler):
                h.setLevel(lvl)
        except Exception:
            continue
