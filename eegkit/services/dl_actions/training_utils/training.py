"""Training loop helpers for DL models."""

from __future__ import annotations

import logging
import time
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
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
    epoch_logger: Optional[Callable[[Dict[str, float]], None]] = None,
) -> nn.Module:
    """Run training with optional early stopping; return model with best val loss."""
    best_val = float("inf")
    best_epoch = -1
    no_improve = 0
    best_state: Optional[Dict] = None

    pbar = tqdm(range(epochs), desc="Training", unit="epoch", dynamic_ncols=True)
    for epoch in pbar:
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0
        train_batches = 0
        for x_b, y_b in tr_dl:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            y_p = model(x_b)
            loss = loss_fn(y_p, y_b)
            loss.backward()

            # --- GRADIENT CHECK SNIPPET ---
            has_gradients = False
            for name, param in model.named_parameters():
                if param.grad is not None:
                    has_gradients = True
                    # Check if gradients are vanishingly small (e.g., all zeros)
                    if param.grad.abs().max() < 1e-6:
                        logger.warning(f"Vanishing gradient in {name} - Max grad: {param.grad.abs().max():.8f}")
                    break # Check only the first trainable layer

            if not has_gradients:
                logger.error("FATAL: No gradients found in the model at all!")
            # ------------------------------

            optimizer.step()
            train_loss += loss.item()
            train_batches += 1

        train_loss /= max(train_batches, 1)

        model.eval()
        val_loss = 0.0
        val_preds = []
        val_trues = []
        with torch.no_grad():
            for x_b, y_b in val_dl:
                x_b, y_b = x_b.to(device), y_b.to(device)
                y_p = model(x_b)
                val_loss += loss_fn(y_p, y_b).item()
                val_preds.extend(y_p.argmax(dim=1).cpu().numpy())
                val_trues.extend(y_b.cpu().numpy())
        val_loss /= max(len(val_dl), 1)
        val_accuracy = float(accuracy_score(val_trues, val_preds))

        if scheduler is not None:
            try:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            except Exception as exc:
                logger.warning("scheduler.step failed at epoch %d: %s", epoch + 1, exc)

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_epoch = epoch + 1
            no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
        
        epoch_elapsed = time.time() - epoch_start_time

        pbar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}", best=f"{best_val:.4f}", improved=improved)
        if epoch_logger is not None:
            try:
                current_lr = optimizer.param_groups[0]["lr"]
                epoch_logger(
                    {
                        "epoch": float(epoch + 1),
                        "train/loss": float(train_loss),
                        "val/loss": float(val_loss),
                        "val/accuracy": float(val_accuracy),
                        "val/best_loss": float(best_val),
                        "val/improved": float(1.0 if improved else 0.0),
                        "train/learning_rate": float(current_lr),
                        "train/epoch_time_sec": float(epoch_elapsed),
                    }
                )
            except Exception as exc:
                logger.warning("epoch logger failed at epoch %d: %s", epoch + 1, exc)

        if es_patience > 0 and no_improve >= es_patience:
            logger.info("early stopping triggered at epoch %d", epoch + 1)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch
