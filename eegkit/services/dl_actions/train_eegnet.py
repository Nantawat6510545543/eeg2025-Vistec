"""Train a binary EEGNet on a saved DL epoch tensor dataset."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from ...models.dtos import BaseTaskDTO, EEGNetBinaryTrainParamsDTO
from ...services.ai_models.deep_learning.EEGNetBinary import EEGNetBinary
from .training_utils import (
    build_dataloaders,
    build_loss_function,
    build_lr_scheduler,
    class_weights,
    compute_binary_metrics,
    eval_loop,
    resolve_balance_config,
    resolve_device_and_seed,
    resolve_early_stopping_patience,
    split_with_groups,
    train_loop,
)
from . import register_dl

logger = logging.getLogger(__name__)


def _parse_seeds_csv(value: object) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out: list[int] = []
        for v in value:
            try:
                out.append(int(v))
            except Exception:
                continue
        return out
    s = str(value).strip()
    if not s:
        return []
    seeds: list[int] = []
    for part in s.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            seeds.append(int(part))
        except Exception:
            continue
    # de-dup, preserve order
    seen: set[int] = set()
    uniq: list[int] = []
    for seed in seeds:
        if seed not in seen:
            seen.add(seed)
            uniq.append(seed)
    return uniq


@register_dl("Train EEGNet", EEGNetBinaryTrainParamsDTO)
def train_eegnet(self, task_dto: BaseTaskDTO, params: EEGNetBinaryTrainParamsDTO):
    """Train a binary EEGNet from a saved DL epoch tensor dataset and return results."""
    selected = self.selected_value(getattr(params, "dataset_path", None))
    task_name = getattr(task_dto, "task", None)
    dataset_path = self.resolve_dataset_path(selected, task_name)
    if not dataset_path:
        return {"status": "no_dataset_selected", "message": "Select a dataset before training."}

    arrays = self.load_xyg_dataset(dataset_path)
    x = np.asarray(arrays["x"], dtype=np.float32)
    y = (np.asarray(arrays["y"]) > 0).astype(np.int64)
    groups = np.asarray(arrays["group"]).astype(str)

    if x.ndim != 4 or x.shape[1] != 1:
        return {"status": "bad_shape", "message": f"Expected x shape (N,1,C,T), got {x.shape}."}
    if len(x) < 10:
        return {"status": "too_few_samples", "message": f"Only {len(x)} samples — need at least 10."}
    if not np.isfinite(x).all():
        return {"status": "bad_values", "message": "Input dataset contains NaN or Inf values."}
    if np.unique(y).size < 2:
        label_counts = dict(zip(*np.unique(y, return_counts=True)))
        return {
            "status": "single_class_dataset",
            "message": f"Dataset has only one class: {label_counts}. Check CCD label extraction.",
        }

    x_mean = x.mean(axis=(2, 3), keepdims=True)
    x_std = x.std(axis=(2, 3), keepdims=True)
    x = (x - x_mean) / np.clip(x_std, 1e-6, None)

    test_split = float(getattr(params, "test_split", 0.25))
    val_split = float(getattr(params, "val_split", 0.25))
    split_seed = int(getattr(params, "seed", 42))

    x_tr, y_tr, x_val, y_val, x_te, y_te = split_with_groups(
        x,
        y,
        groups,
        test_split=test_split,
        val_split=val_split,
        seed=split_seed,
    )
    if np.unique(y_tr).size < 2:
        train_counts = dict(zip(*np.unique(y_tr, return_counts=True)))
        return {
            "status": "single_class_train_split",
            "message": f"Training split has only one class: {train_counts}. Adjust split or inspect labels.",
        }

    n_channels = x_tr.shape[2]
    n_timepoints = x_tr.shape[3]

    # Multi-seed training: keep the split fixed (split_seed) and vary training randomness.
    seeds = _parse_seeds_csv(getattr(params, "seeds", ""))
    if not seeds:
        seeds = [split_seed]

    dataset_name = Path(dataset_path).name
    run_name = str(getattr(params, "run_name", "dl_train"))
    base_model_name = f"{run_name}_{dataset_name}"

    # Config resolved once (shared across seeds)
    balance = resolve_balance_config(params, select_value=self.selected_value)
    lr = float(getattr(params, "lr", 1e-3))
    min_lr = float(getattr(params, "min_lr", 1e-6))
    epochs_n = int(getattr(params, "epochs_n", 50))
    batch_size = int(getattr(params, "batch_size", 32))
    early_stopping = bool(getattr(params, "early_stopping", False))
    es_patience = resolve_early_stopping_patience(params)
    loss_choice = self.selected_value(getattr(params, "loss_function", "cross_entropy"))
    loss_name = str(loss_choice or "cross_entropy")
    label_smoothing = float(getattr(params, "label_smoothing", 0.0))
    scheduler_type_choice = self.selected_value(getattr(params, "scheduler_type", "reduce_on_plateau"))
    scheduler_type = str(scheduler_type_choice or "reduce_on_plateau").strip().lower()

    label_balance_all = dict(zip(*np.unique(y, return_counts=True)))
    label_balance_splits = {
        "train": dict(zip(*np.unique(y_tr, return_counts=True))),
        "val": dict(zip(*np.unique(y_val, return_counts=True))),
        "test": dict(zip(*np.unique(y_te, return_counts=True))),
    }

    # DataLoaders depend on sampling strategy and seed.
    def _build_loaders(seed_for_loader: int):
        return build_dataloaders(
            x_tr,
            y_tr,
            x_val,
            y_val,
            x_te,
            y_te,
            batch_size,
            balance.weighted_sampler,
            undersample=balance.undersample,
            seed=int(seed_for_loader),
        )

    best = {
        "seed": None,
        "score": -float("inf"),
        "model": None,
        "metrics": None,
        "best_epoch": None,
        "device_str": None,
    }
    per_seed: list[Dict[str, Any]] = []

    logger.info(
        "Training EEGNet (multi-seed=%s): n_ch=%d n_t=%d split_seed=%d seeds=%s",
        "yes" if len(seeds) > 1 else "no",
        int(n_channels),
        int(n_timepoints),
        int(split_seed),
        seeds,
    )
    logger.info(
        "Label balance all/train/val/test: %s / %s / %s / %s",
        label_balance_all,
        label_balance_splits["train"],
        label_balance_splits["val"],
        label_balance_splits["test"],
    )

    for seed_for_train in seeds:
        device_str, device = resolve_device_and_seed(
            self.selected_value(getattr(params, "device", "cpu")),
            seed=int(seed_for_train),
        )

        model_name = f"{base_model_name}_seed{int(seed_for_train)}" if len(seeds) > 1 else base_model_name
        model = EEGNetBinary(n_channels=int(n_channels), n_timepoints=int(n_timepoints)).to(device)

        weight_tensor = class_weights(y_tr, device) if balance.use_class_weights else None
        loss_fn = build_loss_function(loss_name, weight_tensor, label_smoothing)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = build_lr_scheduler(
            optimizer,
            params,
            min_lr=min_lr,
            select_value=self.selected_value,
        )

        tr_dl, val_dl, te_dl = _build_loaders(seed_for_loader=seed_for_train)

        # W&B: keep behaviour unchanged; if enabled, each seed becomes a separate run.
        default_wandb_enabled = bool(os.getenv("WANDB_API_KEY"))
        wandb_enabled = bool(getattr(params, "wandb_enabled", default_wandb_enabled))
        wandb_project = str(getattr(params, "wandb_project", os.getenv("WANDB_PROJECT", "ku-final-eegkit")))
        wandb_entity = getattr(params, "wandb_entity", None)
        wandb_tags_raw = getattr(params, "wandb_tags", None)
        wandb_run: Optional[Any] = None
        epoch_logger = None

        if wandb_enabled:
            try:
                import wandb

                init_kwargs: Dict[str, Any] = {
                    "project": wandb_project,
                    "name": model_name,
                    "config": {
                        "dataset_path": dataset_path,
                        "run_name": run_name,
                        "model_name": model_name,
                        "device": device_str,
                        "n_channels": int(n_channels),
                        "n_timepoints": int(n_timepoints),
                        "train_samples": int(len(x_tr)),
                        "val_samples": int(len(x_val)),
                        "test_samples": int(len(x_te)),
                        "loss_function": loss_name,
                        "label_smoothing": label_smoothing,
                        "balance_strategy": balance.strategy,
                        "weighted_sampler": balance.weighted_sampler,
                        "undersample": balance.undersample,
                        "weight_classes": balance.use_class_weights,
                        "lr": lr,
                        "min_lr": min_lr,
                        "scheduler_type": scheduler_type,
                        "scheduler_mode": self.selected_value(getattr(params, "scheduler_mode", "min")),
                        "scheduler_patience": int(getattr(params, "scheduler_patience", 10)),
                        "scheduler_threshold": float(getattr(params, "scheduler_threshold", 1e-4)),
                        "scheduler_threshold_mode": self.selected_value(
                            getattr(params, "scheduler_threshold_mode", "rel")
                        ),
                        "scheduler_cooldown": int(getattr(params, "scheduler_cooldown", 0)),
                        "early_stopping": early_stopping,
                        "early_stopping_patience": int(es_patience),
                        "epochs": epochs_n,
                        "batch_size": batch_size,
                        "seed": int(seed_for_train),
                        "split_seed": int(split_seed),
                        "multi_seed": bool(len(seeds) > 1),
                    },
                }
                if wandb_entity:
                    init_kwargs["entity"] = str(wandb_entity)
                if wandb_tags_raw:
                    if isinstance(wandb_tags_raw, str):
                        tags = [t.strip() for t in wandb_tags_raw.split(",") if t.strip()]
                    elif isinstance(wandb_tags_raw, (list, tuple)):
                        tags = [str(t).strip() for t in wandb_tags_raw if str(t).strip()]
                    else:
                        tags = []
                    if tags:
                        init_kwargs["tags"] = tags

                wandb_run = wandb.init(**init_kwargs)
                epoch_logger = lambda payload: wandb_run.log(payload)
                logger.info("W&B run started: %s", getattr(wandb_run, "name", model_name))
            except Exception as exc:
                logger.warning("W&B disabled due to init failure: %s", exc)

        try:
            model, best_epoch = train_loop(
                model,
                tr_dl,
                val_dl,
                loss_fn,
                optimizer,
                scheduler,
                epochs_n,
                device,
                es_patience,
                epoch_logger=epoch_logger,
            )
            y_true, y_pred, y_prob = eval_loop(model, te_dl, device)
            metrics = compute_binary_metrics(y_true, y_pred, y_prob)

            per_seed.append(
                {
                    "seed": int(seed_for_train),
                    "best_epoch": float(best_epoch) if best_epoch is not None else None,
                    "evaluation": metrics,
                    "device": device_str,
                    "wandb_enabled": wandb_run is not None,
                }
            )

            # Select best by balanced accuracy (fallback to accuracy)
            score = metrics.get("balanced_accuracy")
            if score is None:
                score = metrics.get("accuracy")
            score_f = float(score) if score is not None else float("nan")
            if np.isfinite(score_f) and score_f > float(best["score"]):
                prev_best_model = best.get("model")
                if prev_best_model is not None and prev_best_model is not model:
                    try:
                        del prev_best_model
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                # keep best model
                best.update(
                    {
                        "seed": int(seed_for_train),
                        "score": float(score_f),
                        "model": model,
                        "metrics": metrics,
                        "best_epoch": best_epoch,
                        "device_str": device_str,
                    }
                )
            else:
                # free memory
                try:
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                except Exception:
                    pass

            if wandb_run is not None:
                try:
                    payload: Dict[str, Any] = {
                        "test/accuracy": metrics.get("accuracy"),
                        "test/balanced_accuracy": metrics.get("balanced_accuracy"),
                        "test/f1_binary": metrics.get("f1_binary"),
                        "test/f1_macro": metrics.get("f1_macro"),
                        "test/f1_weighted": metrics.get("f1_weighted"),
                        "test/sensitivity": metrics.get("sensitivity"),
                        "test/specificity": metrics.get("specificity"),
                        "test/roc_auc": metrics.get("roc_auc"),
                        "test/pr_auc": metrics.get("pr_auc"),
                    }
                    if best_epoch is not None:
                        payload["train/best_epoch"] = float(best_epoch)
                    wandb_run.log(payload)
                except Exception as exc:
                    logger.warning("W&B metric logging failed: %s", exc)
        finally:
            if wandb_run is not None:
                try:
                    wandb_run.finish()
                except Exception as exc:
                    logger.warning("W&B finish failed: %s", exc)

    if best["model"] is None:
        return {"status": "no_successful_runs", "message": "No seed run completed successfully."}

    # Aggregate stats across seeds for a few key metrics
    key_metrics = ["accuracy", "balanced_accuracy", "f1_binary", "roc_auc", "pr_auc"]
    aggregate: Dict[str, Any] = {"n_seeds": int(len(seeds)), "seeds": [int(s) for s in seeds]}
    for k in key_metrics:
        vals = []
        for row in per_seed:
            m = row.get("evaluation")
            if isinstance(m, dict) and m.get(k) is not None:
                try:
                    vals.append(float(m.get(k)))
                except Exception:
                    pass
        if vals:
            aggregate[f"{k}_mean"] = float(np.mean(vals))
            aggregate[f"{k}_std"] = float(np.std(vals))
            aggregate[f"{k}_min"] = float(np.min(vals))
            aggregate[f"{k}_max"] = float(np.max(vals))

    best_seed = int(best["seed"]) if best["seed"] is not None else split_seed
    final_model_name = f"{base_model_name}_seed{best_seed}" if len(seeds) > 1 else base_model_name

    return {
        "model": best["model"],
        "model_name": final_model_name,
        "evaluation": best["metrics"],
        "plots": {},
        "multi_seed": {
            "split_seed": int(split_seed),
            "seeds": [int(s) for s in seeds],
            "aggregate": aggregate,
            "runs": per_seed,
            "best_seed": int(best_seed),
            "best_score": float(best["score"]),
            "selection_metric": "balanced_accuracy",
        },
        "metadata": {
            "model_name": final_model_name,
            "dataset_path": dataset_path,
            "n_channels": int(n_channels),
            "n_timepoints": int(n_timepoints),
            "device": str(best.get("device_str") or "unknown"),
            "loss_function": loss_name,
            "label_smoothing": float(label_smoothing),
            "balance_strategy": balance.strategy,
            "weight_classes": balance.use_class_weights,
            "weighted_sampler": balance.weighted_sampler,
            "undersample": balance.undersample,
            "train_samples": int(len(x_tr)),
            "val_samples": int(len(x_val)),
            "test_samples": int(len(x_te)),
            "run_name": run_name,
            "wandb_enabled": any(bool(r.get("wandb_enabled")) for r in per_seed),
            "split_seed": int(split_seed),
            "seeds": [int(s) for s in seeds],
        },
    }
