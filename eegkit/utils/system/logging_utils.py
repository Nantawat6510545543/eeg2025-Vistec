"""Lightweight logging configuration helpers for notebooks and apps."""
from __future__ import annotations

import logging
import os
from typing import Optional


def configure_logging(level: Optional[str] = None) -> None:
    """Configure root logging once with a simple console formatter.

    The level defaults to the LOG_LEVEL environment variable or INFO when unset.
    Subsequent calls are no-ops if handlers are already configured.
    """
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
    """Raise the level of console StreamHandlers to reduce notebook noise.

    Only stream handlers are adjusted; file or other handlers are left unchanged.
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
