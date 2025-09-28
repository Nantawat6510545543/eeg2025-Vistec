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
