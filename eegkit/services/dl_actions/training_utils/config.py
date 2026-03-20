"""Shared config helpers for DL action modules.

Keep trainer functions short by centralizing common parameter resolution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BalanceConfig:
    strategy: str
    weighted_sampler: bool
    undersample: bool
    use_class_weights: bool


def resolve_balance_config(params: Any, *, select_value: Callable[[Any], Any]) -> BalanceConfig:
    """Resolve class-imbalance handling from params.

    Expected param:
      - balance_strategy: one of {"none", "class_weight", "weighted_sampler", "undersample"}

    Invalid values fall back to "class_weight".
    """

    choice = select_value(getattr(params, "balance_strategy", "class_weight"))
    strategy = str(choice or "class_weight").strip().lower()
    if strategy not in {"none", "class_weight", "weighted_sampler", "undersample"}:
        logger.warning("Invalid balance_strategy=%r; falling back to 'class_weight'", strategy)
        strategy = "class_weight"

    return BalanceConfig(
        strategy=strategy,
        weighted_sampler=strategy == "weighted_sampler",
        undersample=strategy == "undersample",
        use_class_weights=strategy == "class_weight",
    )


def resolve_early_stopping_patience(params: Any) -> int:
    """Return early-stopping patience, or 0 if disabled."""

    early_stopping = bool(getattr(params, "early_stopping", False))
    if not early_stopping:
        return 0
    return max(int(getattr(params, "early_stopping_patience", 20)), 1)


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    params: Any,
    *,
    min_lr: float,
    select_value: Callable[[Any], Any],
    patience: Optional[int] = None,
) -> Optional[torch.optim.lr_scheduler.ReduceLROnPlateau]:
    """Build LR scheduler from params.

    Currently supports:
      - scheduler_type: "reduce_on_plateau" | "none"

    Returns:
      Scheduler instance or None.
    """

    scheduler_type_choice = select_value(getattr(params, "scheduler_type", "reduce_on_plateau"))
    scheduler_type = str(scheduler_type_choice or "reduce_on_plateau").strip().lower()
    if scheduler_type == "none":
        return None

    mode_choice = select_value(getattr(params, "scheduler_mode", "min"))
    scheduler_mode = str(mode_choice or "min").strip().lower()

    threshold_mode_choice = select_value(getattr(params, "scheduler_threshold_mode", "rel"))
    threshold_mode = str(threshold_mode_choice or "rel").strip().lower()

    if patience is None:
        scheduler_patience = max(int(getattr(params, "scheduler_patience", 10)), 1)
    else:
        scheduler_patience = max(int(patience), 1)

    factor = float(getattr(params, "lr_factor", 0.5))
    if not (0.0 < factor < 1.0):
        logger.warning("Invalid lr_factor=%s for ReduceLROnPlateau; using 0.5", factor)
        factor = 0.5

    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=scheduler_mode,
        factor=factor,
        patience=scheduler_patience,
        threshold=float(getattr(params, "scheduler_threshold", 1e-4)),
        threshold_mode=threshold_mode,
        cooldown=int(getattr(params, "scheduler_cooldown", 0)),
        min_lr=float(min_lr),
    )
