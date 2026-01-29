"""Model-specific training action for EEGNetMultiReg (regression)."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict
from typing import Any
import sys

import numpy as np

from ...models.dtos import BaseTaskDTO, EEGNetMultiRegTrainParamsDTO
from . import register_ai

logger = logging.getLogger(__name__)


def to_json(data: dict[str, Any]) -> str:
    """Consistent formatted JSON output."""
    return json.dumps(data, indent=2, ensure_ascii=False, default=str)


@register_ai("Train EEGNetMultiReg", EEGNetMultiRegTrainParamsDTO)
def train_eegnet_multireg(self, task_dto: BaseTaskDTO, params: EEGNetMultiRegTrainParamsDTO) -> dict[str, Any]:
    """Train EEGNetMultiReg on selected regression targets and return metrics."""

    t0 = time.perf_counter()

    try:
        import torch
    except Exception:
        return to_json({"status": "error", "reason": "torch not available"})

    # Build dataset via AI service target routing
    X, y_raw, meta = self._build_epoch_dataset(task_dto, params)
    if X is None or y_raw is None:
        return to_json({
            "status": "unavailable",
            "reason": meta.get("reason", "dataset_unavailable")
        })

    if not isinstance(y_raw, np.ndarray):
        return to_json({
            "status": "invalid_target",
            "reason": "expected numeric regression targets"
        })

    y = y_raw.astype(np.float32)

    if y.ndim == 1:
        y = y.reshape(-1, 1)

    if X.shape[0] != y.shape[0]:
        return to_json({
            "status": "shape_mismatch",
            "reason": f"X={X.shape[0]} y={y.shape[0]}"
        })

    # Drop non-finite rows
    mask = np.isfinite(y).all(axis=1)
    dropped = int((~mask).sum())

    if dropped:
        X = X[mask]
        y = y[mask]
        logger.warning("Dropped %d samples with non-finite regression targets.", dropped)

    if X.shape[0] < 2:
        return to_json({
            "status": "too_small",
            "n_samples": int(X.shape[0])
        })

    # Train/validation/test split (epoch-wise for single subject, person-wise for cohort)
    val_split = max(0.0, min(0.9, float(params.val_split)))
    test_split = max(0.0, min(0.9, float(getattr(params, "test_split", 0.2))))
    rng = np.random.default_rng(int(params.seed))

    subj_ids = meta.get("subject_ids")
    if isinstance(subj_ids, (list, np.ndarray)) and len(set(subj_ids)) > 1:
        # Cohort split by subject id
        uniq = list({str(s) for s in subj_ids})
        rng.shuffle(uniq)
        n_subj = len(uniq)
        n_val = int(round(val_split * n_subj)) if val_split > 0 else 0
        n_test = int(round(test_split * n_subj)) if test_split > 0 else 0
        val_subjects = set(uniq[:n_val])
        test_subjects = set(uniq[n_val:n_val + n_test])
        train_subjects = set(uniq[n_val + n_test:])
        subj_ids_arr = np.array([str(s) for s in subj_ids])
        train_idx = np.nonzero(np.isin(subj_ids_arr, list(train_subjects)))[0]
        val_idx = np.nonzero(np.isin(subj_ids_arr, list(val_subjects)))[0]
        test_idx = np.nonzero(np.isin(subj_ids_arr, list(test_subjects)))[0]
    else:
        # Single subject: split epochs
        n = X.shape[0]
        indices = np.arange(n)
        rng.shuffle(indices)
        n_test = int(round(test_split * n)) if test_split > 0 else 0
        n_val = int(round(val_split * n)) if val_split > 0 else 0
        test_idx = indices[:n_test]
        val_idx = indices[n_test:n_test + n_val]
        train_idx = indices[n_test + n_val:]

    if train_idx.size == 0:
        return to_json({"status": "split_error", "reason": "empty train set after splitting"})

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = (X[val_idx], y[val_idx]) if val_idx.size > 0 else (None, None)
    X_test, y_test = (X[test_idx], y[test_idx]) if test_idx.size > 0 else (None, None)

    try:
        import torch
    except Exception:
        return to_json({"status": "error", "reason": "torch not available"})

    device = torch.device(
        "cuda" if torch.cuda.is_available() and (params.device and params.device[0] != "cpu") else "cpu"
    )

    from ..ai_models import EEGNetMultiReg
    model = EEGNetMultiReg(n_outputs=y.shape[1]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=float(params.lr))
    criterion = torch.nn.MSELoss()

    def make_loader(Xa: np.ndarray, ya: np.ndarray):
        Xt = torch.from_numpy(Xa)
        yt = torch.from_numpy(ya)
        ds = torch.utils.data.TensorDataset(Xt, yt)
        return torch.utils.data.DataLoader(ds, batch_size=int(params.batch_size), shuffle=True)

    dl_train = make_loader(X_train, y_train)
    dl_val = make_loader(X_val, y_val) if X_val is not None else None
    dl_test = make_loader(X_test, y_test) if X_test is not None else None

    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_state = None
    patience = int(params.patience)
    no_improve = 0
    epochs_run = 0

    # Progress bar setup
    try:
        from tqdm.auto import tqdm
        interactive = sys.stdout.isatty() or sys.stderr.isatty()
        epoch_iter = tqdm(range(int(params.epochs_n)), desc="Epochs", unit="epoch", disable=not interactive)
    except Exception:
        epoch_iter = range(int(params.epochs_n))

    for epoch in epoch_iter:
        model.train()
        train_loss_accum = 0.0
        batches = 0

        try:
            from tqdm.auto import tqdm as _tqdm
            batch_iter = _tqdm(dl_train, desc=f"Train {epoch+1}", unit="batch",
                               leave=False, disable=not (sys.stdout.isatty() or sys.stderr.isatty()))
        except Exception:
            batch_iter = dl_train

        for xb, yb in batch_iter:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optim.step()
            train_loss_accum += float(loss.item())
            batches += 1

        train_loss = train_loss_accum / max(1, batches)

        val_loss = None
        val_mae = None
        val_r2 = None

        if dl_val is not None:
            model.eval()
            preds_all = []
            y_all = []
            val_batches = 0
            val_loss_sum = 0.0

            try:
                from tqdm.auto import tqdm as _tqdm
                val_iter = _tqdm(dl_val, desc=f"Val {epoch+1}", unit="batch",
                                 leave=False, disable=not (sys.stdout.isatty() or sys.stderr.isatty()))
            except Exception:
                val_iter = dl_val

            with torch.no_grad():
                for xb, yb in val_iter:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    loss_v = criterion(out, yb)
                    val_loss_sum += float(loss_v.item())
                    preds_all.append(out.cpu().numpy())
                    y_all.append(yb.cpu().numpy())
                    val_batches += 1

            val_loss = val_loss_sum / max(1, val_batches)

            try:
                Yp = np.concatenate(preds_all, axis=0)
                Yt = np.concatenate(y_all, axis=0)
                val_mae = float(np.nanmean(np.abs(Yp - Yt)))

                ss_res = np.nansum((Yt - Yp) ** 2, axis=0)
                y_mean = np.nanmean(Yt, axis=0)
                ss_tot = np.nansum((Yt - y_mean) ** 2, axis=0)

                with np.errstate(invalid="ignore", divide="ignore"):
                    r2_each = 1.0 - (ss_res / np.where(ss_tot == 0.0, np.nan, ss_tot))

                val_r2 = float(np.nanmean(r2_each))

            except Exception:
                val_mae = None
                val_r2 = None

            # Early stopping
            if val_loss < best_loss - 1e-9:
                best_loss = val_loss
                best_state = {
                    "model_state": model.state_dict(),
                    "optim_state": optim.state_dict(),
                }
                no_improve = 0
            else:
                no_improve += 1
                if patience > 0 and no_improve >= patience:
                    epochs_run = epoch + 1
                    history.append({"epoch": epoch+1, "train_loss": train_loss,
                                    "val_loss": val_loss, "val_mae": val_mae, "val_r2": val_r2})
                    break

        # Evaluate train and test metrics for reporting
        train_mae = None
        train_r2 = None
        try:
            model.eval()
            preds_all = []
            y_all = []
            with torch.no_grad():
                for xb, yb in dl_train:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    preds_all.append(out.cpu().numpy())
                    y_all.append(yb.cpu().numpy())
            if preds_all:
                Yp = np.concatenate(preds_all, axis=0)
                Yt = np.concatenate(y_all, axis=0)
                train_mae = float(np.nanmean(np.abs(Yp - Yt)))
                ss_res = np.nansum((Yt - Yp) ** 2, axis=0)
                y_mean = np.nanmean(Yt, axis=0)
                ss_tot = np.nansum((Yt - y_mean) ** 2, axis=0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    r2_each = 1.0 - (ss_res / np.where(ss_tot == 0.0, np.nan, ss_tot))
                train_r2 = float(np.nanmean(r2_each))
        except Exception:
            train_mae = None
            train_r2 = None

        test_mae = None
        test_r2 = None
        if dl_test is not None:
            try:
                model.eval()
                preds_all = []
                y_all = []
                with torch.no_grad():
                    for xb, yb in dl_test:
                        xb, yb = xb.to(device), yb.to(device)
                        out = model(xb)
                        preds_all.append(out.cpu().numpy())
                        y_all.append(yb.cpu().numpy())
                if preds_all:
                    Yp = np.concatenate(preds_all, axis=0)
                    Yt = np.concatenate(y_all, axis=0)
                    test_mae = float(np.nanmean(np.abs(Yp - Yt)))
                    ss_res = np.nansum((Yt - Yp) ** 2, axis=0)
                    y_mean = np.nanmean(Yt, axis=0)
                    ss_tot = np.nansum((Yt - y_mean) ** 2, axis=0)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        r2_each = 1.0 - (ss_res / np.where(ss_tot == 0.0, np.nan, ss_tot))
                    test_r2 = float(np.nanmean(r2_each))
            except Exception:
                test_mae = None
                test_r2 = None

        # Print epoch-style summary similar to the reference image, adapted to regression
        try:
            print(f"\nEpoch  {epoch}")
            print(['mae', 'r2'])
            print("Training Loss", train_loss)
            print("Train - ", [train_mae, train_r2])
            print("Validation - ", [val_mae, val_r2])
            print("Test - ", [test_mae, test_r2])
        except Exception:
            pass

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae": val_mae,
            "val_r2": val_r2
        })
        epochs_run = epoch + 1

    if best_state is None:
        best_state = {
            "model_state": model.state_dict(),
            "optim_state": optim.state_dict()
        }

    checkpoint = None
    if params.save_checkpoint:
        digest_items = (
            getattr(task_dto, "subject", None),
            getattr(task_dto, "task", None),
            getattr(task_dto, "run", None),
            y.shape[1], params.lr, params.batch_size, params.epochs_n,
            params.val_split, params.seed
        )
        digest = hashlib.sha1(repr(digest_items).encode()).hexdigest()[:10]
        root = os.path.join("jobs", "ai", "train", "EEGNetMultiReg", digest)
        os.makedirs(root, exist_ok=True)

        model_path = os.path.join(root, "model.pt")
        torch.save(best_state["model_state"], model_path)

        meta_path = os.path.join(root, "meta.json")
        meta_out = {
            "digest": digest,
            "history": history,
            "best_val_loss": best_loss if val_split > 0 else None,
            "params": asdict(params),
            "dataset_shape": X.shape,
            "targets": int(y.shape[1]),
            "target_cols": meta.get("target_cols"),
            "split": {
                "val_split": float(val_split),
                "test_split": float(test_split),
                "cohort_split": bool(isinstance(subj_ids, (list, np.ndarray)) and len(set(subj_ids)) > 1),
            },
            "duration_sec": round(time.perf_counter() - t0, 4),
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_out, f, indent=2, ensure_ascii=False, default=str)

        checkpoint = model_path

    last = history[-1] if history else {}
    result = {
        "status": "ok",
        "regression": True,
        "targets": int(y.shape[1]),
        "target_names": meta.get("target_cols"),
        "history": history,
        "epochs_run": epochs_run,
        "best_val_loss": best_loss if val_split > 0 else None,
        "last_val_mae": last.get("val_mae"),
        "last_val_r2": last.get("val_r2"),
        "checkpoint": checkpoint,
        "dataset_shape": X.shape,
        "duration_sec": round(time.perf_counter() - t0, 4),
    }

    return to_json(result)
