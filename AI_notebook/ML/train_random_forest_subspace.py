"""Train Random Forest regressors with random subspace and save as a pickle."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def fit_single_target_model(X_tr: np.ndarray, y_tr: np.ndarray, groups_tr: np.ndarray) -> GridSearchCV:
    """Fit one target model using Random Forest with random subspace."""
    single_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("rf", RandomForestRegressor(random_state=42, n_jobs=-1)),
        ]
    )
    single_grid = {
        "rf__n_estimators": [50, 100, 200],
        "rf__max_features": [0.3, 0.5, 0.7, "sqrt", "log2"],
        "rf__max_depth": [10, 20, 30, None],
        "rf__min_samples_split": [2, 5, 10],
        "rf__min_samples_leaf": [1, 2, 4],
    }
    search = GridSearchCV(
        estimator=single_pipe,
        param_grid=single_grid,
        scoring="neg_mean_absolute_error",
        cv=GroupKFold(n_splits=4),
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    search.fit(X_tr, y_tr, groups=groups_tr)
    return search


def summarize(y_true: np.ndarray, y_hat: np.ndarray) -> dict:
    """Compute simple ACC/RT metrics."""
    return {
        "acc_corr": float(np.corrcoef(y_true[:, 0], y_hat[:, 0])[0, 1]),
        "rt_corr": float(np.corrcoef(y_true[:, 1], y_hat[:, 1])[0, 1]),
        "acc_mae": float(mean_absolute_error(y_true[:, 0], y_hat[:, 0])),
        "rt_mae": float(mean_absolute_error(y_true[:, 1], y_hat[:, 1])),
    }


def main() -> int:
    """Run training and save the model artifact."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelname", default="random_forest_subspace")
    parser.add_argument("--n-subjects", type=int, default=500)
    parser.add_argument(
        "--data-dir",
        default="/mount/NAS-workspace-portal/eeg2025-Vistec/models/data",
    )
    parser.add_argument(
        "--model-dir",
        default="/mount/NAS-workspace-portal/eeg2025-Vistec/models/ml",
    )
    parser.add_argument("--x-file", default="")
    parser.add_argument("--y-file", default="")
    parser.add_argument("--groups-file", default="")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    x_file = Path(args.x_file) if args.x_file else sorted(data_dir.glob("X_*.npy"))[0]
    y_file = Path(args.y_file) if args.y_file else sorted(data_dir.glob("Y_*.npy"))[0]

    X = np.load(x_file)
    y = np.load(y_file)

    if args.groups_file:
        groups = np.load(args.groups_file)
    else:
        groups = np.arange(len(X))

    n = min(len(X), len(y), len(groups), args.n_subjects)
    X = X[:n]
    y = y[:n]
    groups = groups[:n]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]

    y_train_acc = y_train[:, 0]
    y_train_rt = y_train[:, 1]

    acc_search = fit_single_target_model(X_train, y_train_acc, groups_train)
    rt_search = fit_single_target_model(X_train, y_train_rt, groups_train)

    acc_pred = acc_search.best_estimator_.predict(X_test)
    rt_pred = rt_search.best_estimator_.predict(X_test)
    y_pred_two = np.column_stack([acc_pred, rt_pred])

    model_path = Path(
        f"/mount/NAS-workspace-portal/eeg2025-Vistec/models/ml/{args.modelname}_{args.n_subjects}_{int(time.time())}.pkl"
    )

    artifact = {
        "modelname": args.modelname,
        "n_subjects": args.n_subjects,
        "x_file": str(x_file),
        "y_file": str(y_file),
        "acc_model": acc_search.best_estimator_,
        "rt_model": rt_search.best_estimator_,
        "acc_best_params": acc_search.best_params_,
        "rt_best_params": rt_search.best_params_,
        "metrics": summarize(y_test, y_pred_two),
    }
    joblib.dump(artifact, model_path)

    print("Saved model:", model_path)
    print("ACC params:", acc_search.best_params_)
    print("RT params:", rt_search.best_params_)
    print("Metrics:", artifact["metrics"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
