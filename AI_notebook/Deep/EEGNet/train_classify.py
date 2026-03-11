"""Train a binary EEG model from notebook-exported data."""

import argparse
import csv
import importlib.util
import json
import os
import random
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import WeightedRandomSampler

CWD = os.path.dirname(os.path.abspath(__file__))
if CWD not in sys.path:
    sys.path.append(CWD)

from TrainerClassify import TrainerClassify


def _load_cache_utils_module():
    """Load local data_cache_utils module from this folder."""
    module_path = os.path.join(CWD, "data_cache_utils.py")
    spec = importlib.util.spec_from_file_location("data_cache_utils", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load data_cache_utils from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def str2bool(value):
    """Convert a value to boolean."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def write_csv_row(path: str, data, mode: str):
    """Write one row to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode, newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(list(data))


class NumpyDataset(TorchDataset):
    """Wrap numpy arrays as a torch dataset."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.as_tensor(self.x[idx], dtype=torch.float32), torch.as_tensor(self.y[idx], dtype=torch.long)


class Conv2dWithConstraint(nn.Conv2d):
    """Apply max-norm constraint to conv weights during forward."""

    def __init__(self, *args, max_norm=1.0, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class EEGNet(nn.Module):
    """Define EEGNet model for binary EEG classification."""

    def __init__(
        self,
        Chans=64,
        Samples=128,
        dropoutRate=0.5,
        kernLength=64,
        F1=8,
        D=2,
        F2=16,
        norm_rate=0.25,
        dropoutType="Dropout",
    ):
        super().__init__()
        _ = norm_rate

        if dropoutType == "SpatialDropout2D":
            dropout_layer = nn.Dropout2d(p=dropoutRate)
        else:
            dropout_layer = nn.Dropout(p=dropoutRate)

        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernLength), padding=(0, kernLength // 2), bias=False),
            nn.BatchNorm2d(F1),
            Conv2dWithConstraint(F1, F1 * D, (Chans, 1), groups=F1, bias=False, max_norm=1.0),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            dropout_layer,
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            dropout_layer,
        )

        self.flatten_size = F2 * max(1, Samples // 4 // 8)
        self.classifier = nn.Linear(self.flatten_size, 2)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


@dataclass
class RunConfig:
    """Store run hyperparameters."""

    lr: float
    min_lr: float
    batch_size: int
    patience: int
    epochs: int
    es_patience: int
    factor: float


class FocalLoss(nn.Module):
    """Compute focal loss for binary classification logits with 2 output units."""

    def __init__(self, gamma: float = 2.0, alpha: float = 1.0, class_weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.class_weight = class_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, target, reduction="none", weight=self.class_weight)
        pt = torch.exp(-ce)
        focal = self.alpha * (1.0 - pt) ** self.gamma * ce
        return focal.mean()


def set_determinism(seed: int):
    """Set reproducibility controls."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_cached_xyg(cache_dir: str, params_json: str, n_subjects: int, cache_key: str = ""):
    """Load cached X, y, groups arrays using params-based key."""
    cache_utils = _load_cache_utils_module()
    build_cache_key = cache_utils.build_cache_key
    load_cache = cache_utils.load_cache

    if cache_key:
        resolved_key = cache_key
    else:
        if not os.path.exists(params_json):
            raise FileNotFoundError(f"params_json not found: {params_json}")
        with open(params_json, "r", encoding="utf-8") as file:
            params_obj = json.load(file)
        resolved_key = build_cache_key(params_obj=params_obj, n_subjects=n_subjects, key_prefix="ccd_eegnet")

    X, y, groups, _responds, _paths = load_cache(cache_dir=cache_dir, cache_key=resolved_key, require_responds=False)
    if X is None:
        raise FileNotFoundError(
            f"No cache found for key '{resolved_key}'. "
            "Build cache in notebook first (X/y/groups)."
        )

    y = (np.asarray(y) > 0).astype(np.int64)
    groups = np.asarray(groups).astype(str)

    if X.ndim != 4:
        raise AssertionError(f"Expected cached X shape (S*N, D, C, T). Got {X.shape}")
    if X.shape[1] != 1:
        raise AssertionError(f"Expected depth axis D=1 for EEGNet. Got D={X.shape[1]}")
    if y.ndim != 1:
        raise AssertionError(f"Expected y shape (S*N,), got {y.shape}")
    if groups.ndim != 1:
        raise AssertionError(f"Expected groups shape (S*N,), got {groups.shape}")
    if not (len(X) == len(y) == len(groups)):
        raise AssertionError(
            f"Length mismatch: len(X)={len(X)}, len(y)={len(y)}, len(groups)={len(groups)}"
        )

    return X, y, groups, resolved_key


def split_group_50_25_25(X, y, groups):
    """Create subject-independent 50/25/25 split from X, y, groups."""
    gss_1 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    train_idx, temp_idx = next(gss_1.split(X, y, groups=groups))

    X_train = X[train_idx].astype(np.float32, copy=False)
    y_train = y[train_idx].astype(np.int64, copy=False)

    X_temp = X[temp_idx].astype(np.float32, copy=False)
    y_temp = y[temp_idx].astype(np.int64, copy=False)
    groups_temp = groups[temp_idx]

    gss_2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=43)
    valid_idx_rel, test_idx_rel = next(gss_2.split(X_temp, y_temp, groups=groups_temp))

    X_val = X_temp[valid_idx_rel]
    y_val = y_temp[valid_idx_rel]
    X_test = X_temp[test_idx_rel]
    y_test = y_temp[test_idx_rel]

    for name, arr in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
        if arr.ndim != 4:
            raise AssertionError(f"{name} must be (N, D, C, T), got {arr.shape}")
        if arr.shape[1] != 1:
            raise AssertionError(f"{name} must have D=1 for EEGNet, got D={arr.shape[1]}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def parse_args():
    """Parse command line arguments with optional JSON config defaults."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="")
    pre_args, _ = pre_parser.parse_known_args()

    config = {}
    if pre_args.config:
        if not os.path.exists(pre_args.config):
            raise FileNotFoundError(f"config file not found: {pre_args.config}")
        with open(pre_args.config, "r", encoding="utf-8") as file:
            config = json.load(file)
        if not isinstance(config, dict):
            raise ValueError("config file must contain a JSON object")

    parser = argparse.ArgumentParser(description="Train binary EEG model from cached X/y/groups.")
    parser.add_argument("--config", type=str, default=pre_args.config)
    parser.add_argument(
        "--cache_dir", type=str, default=config.get("cache_dir", "/mount/NAS-workspace-portal/eeg2025-Vistec/models/data")
    )
    parser.add_argument("--params_json", type=str, default=config.get("params_json", ""))
    parser.add_argument("--n_subjects", type=int, default=config.get("n_subjects"), required="n_subjects" not in config)
    parser.add_argument("--cache_key", type=str, default=config.get("cache_key", ""))
    parser.add_argument(
        "--output_root", type=str, default=config.get("output_root", "/mount/NAS-workspace-portal/eeg2025-Vistec/jobs")
    )
    parser.add_argument("--run_name", type=str, default=config.get("run_name", "notebook_binary_run"))
    parser.add_argument("--cuda_device", type=str, default=config.get("cuda_device", "cpu"))

    parser.add_argument("--lr", type=float, default=config.get("lr", 1e-3))
    parser.add_argument("--min_lr", type=float, default=config.get("min_lr", 1e-6))
    parser.add_argument("--batch_size", type=int, default=config.get("batch_size", 64))
    parser.add_argument("--patience", type=int, default=config.get("patience", 10))
    parser.add_argument("--epochs", type=int, default=config.get("epochs", 100))
    parser.add_argument("--es_patience", type=int, default=config.get("es_patience", 20))
    parser.add_argument("--factor", type=float, default=config.get("factor", 0.5))
    parser.add_argument("--early_stopping_enabled", type=str2bool, default=config.get("early_stopping_enabled", False))
    parser.add_argument("--inference_only", type=str2bool, default=config.get("inference_only", False))
    parser.add_argument("--loss_type", type=str, choices=["ce", "focal"], default=config.get("loss_type", "ce"))
    parser.add_argument("--focal_gamma", type=float, default=config.get("focal_gamma", 2.0))
    parser.add_argument("--focal_alpha", type=float, default=config.get("focal_alpha", 1.0))
    parser.add_argument("--class_weights_enabled", type=str2bool, default=config.get("class_weights_enabled", True))
    parser.add_argument(
        "--weighted_sampler_enabled", type=str2bool, default=config.get("weighted_sampler_enabled", True)
    )

    parser.add_argument("--seed", type=int, default=config.get("seed", 42))
    return parser.parse_args()


def build_dataloaders(x_train, y_train, x_val, y_val, x_test, y_test, batch_size, weighted_sampler_enabled=False):
    """Build train/val/test dataloaders."""
    train_ds = NumpyDataset(x_train, y_train)
    val_ds = NumpyDataset(x_val, y_val)
    test_ds = NumpyDataset(x_test, y_test)

    sampler = None
    shuffle = True
    if weighted_sampler_enabled:
        class_counts = np.bincount(y_train, minlength=2)
        class_counts = np.maximum(class_counts, 1)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[y_train]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_dl, val_dl, test_dl


def build_class_weight_tensor(y_train: np.ndarray, device: str, enabled: bool = True):
    """Build balanced class weights tensor from train labels."""
    class_counts = np.bincount(y_train, minlength=2)
    if not enabled:
        return None, class_counts, None

    classes = np.array([0, 1], dtype=np.int64)
    if np.all(class_counts > 0):
        class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train).astype(np.float32)
    else:
        safe_counts = np.maximum(class_counts, 1)
        class_weights = (safe_counts.sum() / (2.0 * safe_counts)).astype(np.float32)

    class_weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    return class_weight_tensor, class_counts, class_weights


def main():
    """Run end-to-end binary training using notebook data and user model/params."""
    args = parse_args()

    if args.cuda_device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable. Falling back to CPU.")
        args.cuda_device = "cpu"

    set_determinism(args.seed)

    X, y, groups, resolved_key = load_cached_xyg(
        cache_dir=args.cache_dir,
        params_json=args.params_json,
        n_subjects=args.n_subjects,
        cache_key=args.cache_key,
    )
    x_train, y_train, x_val, y_val, x_test, y_test = split_group_50_25_25(X, y, groups)
    print(
        f"Loaded cache key: {resolved_key} | X(S*N,D,C,T)={X.shape} | "
        f"train/val/test = {len(x_train)}/{len(x_val)}/{len(x_test)}"
    )
    n_channels = x_train.shape[2]
    n_timepoints = x_train.shape[3]

    model = EEGNet(
        Chans=n_channels,
        Samples=n_timepoints,
        dropoutRate=0.5,
        kernLength=64,
        F1=8,
        D=2,
        F2=16,
        norm_rate=0.25,
        dropoutType="Dropout",
    )
    model = model.to(args.cuda_device)

    cfg = RunConfig(
        lr=args.lr,
        min_lr=args.min_lr,
        batch_size=args.batch_size,
        patience=args.patience,
        epochs=args.epochs,
        es_patience=args.es_patience,
        factor=args.factor,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        "min",
        factor=cfg.factor,
        patience=cfg.patience,
        min_lr=cfg.min_lr,
    )
    class_weight_tensor, class_counts, class_weights_np = build_class_weight_tensor(
        y_train=y_train,
        device=args.cuda_device,
        enabled=args.class_weights_enabled,
    )

    if args.loss_type == "focal":
        loss_fn = FocalLoss(
            gamma=args.focal_gamma,
            alpha=args.focal_alpha,
            class_weight=class_weight_tensor,
        )
    else:
        loss_fn = nn.CrossEntropyLoss(weight=class_weight_tensor)

    save_dir = os.path.join(args.output_root, args.run_name)
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "model.pt")
    csv_path = os.path.join(save_dir, "results.csv")

    train_dl, val_dl, test_dl = build_dataloaders(
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        cfg.batch_size,
        weighted_sampler_enabled=args.weighted_sampler_enabled,
    )

    print(
        f"loss_type={args.loss_type}, class_weights_enabled={args.class_weights_enabled}, "
        f"weighted_sampler_enabled={args.weighted_sampler_enabled}, class_counts={class_counts.tolist()}, "
        f"class_weights={(class_weights_np.tolist() if class_weights_np is not None else None)}"
    )

    trainer = TrainerClassify(
        model=model,
        epochs=cfg.epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        es_patience=cfg.es_patience,
        batch_size=cfg.batch_size,
        directory=model_path,
        device=torch.device(args.cuda_device),
        early_stopping_enabled=args.early_stopping_enabled,
    )

    if not args.inference_only:
        trainer.train_classify(train_loader=train_dl, val_loader=val_dl)

    test_loss, test_f1, y_pred, y_pred_prob = trainer.eval_classify(test_dl)
    metrics = trainer.eval_result(test_loss, test_f1, y_pred, y_test, y_pred_prob=y_pred_prob)[1]

    write_csv_row(csv_path, metrics.keys(), mode="w")
    write_csv_row(csv_path, metrics.values(), mode="a")
    print(f"Done. Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
