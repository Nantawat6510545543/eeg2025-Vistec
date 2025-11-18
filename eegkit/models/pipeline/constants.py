"""Pipeline-level event constants and mappings.

These are shared across preprocessors/tasks and separated from implementation
to keep `task_processor` focused on orchestration.
"""
from __future__ import annotations

import itertools

# Fixed mapping for surround suppression task
BACKGROUND = [0, 1]
FOREGROUND = [0.0, 0.3, 0.6, 1.0]
STIM = [1, 2, 3]

EVENT_ID = {
    f"bg{b}_fg{f:.1f}_stim{s}": i + 1
    for i, (b, f, s) in enumerate(itertools.product(BACKGROUND, FOREGROUND, STIM))
}


# Resting-state events
RESTING_STATE_EVENT_ID = {
    'open': 1,
    'close': 2,
}


# Contrast Change Detection (CCD) events
CCD_EVENT_ID = {
    'trial_start': 1,
}

__all__ = [
    "BACKGROUND",
    "FOREGROUND",
    "STIM",
    "EVENT_ID",
    "RESTING_STATE_EVENT_ID",
    "CCD_EVENT_ID",
]
