"""Training loop helpers for DL models."""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def train_loop(
    model: nn.Module,
    tr_dl: DataLoader,
    val_dl: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epochs: int,
    device: torch.device,
    es_patience: int,
) -> nn.Module:
    """Run training with optional early stopping; return model with best val loss."""
    best_val = float("inf")
    no_improve = 0
    best_state: Optional[Dict] = None

    pbar = tqdm(range(epochs), desc="Training", unit="epoch", dynamic_ncols=True)
    for epoch in pbar:
        model.train()
        for x_b, y_b in tr_dl:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss_fn(model(x_b), y_b).backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_b, y_b in val_dl:
                x_b, y_b = x_b.to(device), y_b.to(device)
                val_loss += loss_fn(model(x_b), y_b).item()
        val_loss /= max(len(val_dl), 1)
        scheduler.step(val_loss)

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        pbar.set_postfix(val_loss=f"{val_loss:.4f}", best=f"{best_val:.4f}", improved=improved)
        logger.debug("epoch %d/%d  val_loss=%.4f", epoch + 1, epochs, val_loss)

        if es_patience > 0 and no_improve >= es_patience:
            logger.info("early stopping triggered at epoch %d", epoch + 1)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model
