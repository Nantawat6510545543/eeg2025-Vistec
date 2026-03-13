"""Plot helpers for DL training diagnostics."""

from __future__ import annotations

import numpy as np


def build_shaded_error_bar_plot(x_te: np.ndarray, y_te: np.ndarray, model_name: str):
    """Plot mean +/- SEM envelope over time for each true class on test samples."""
    import matplotlib.pyplot as plt

    x_te = np.asarray(x_te)
    y_te = np.asarray(y_te).astype(int)

    if x_te.ndim != 4 or x_te.shape[0] == 0:
        return None

    signals = x_te[:, 0].mean(axis=1)
    time = np.arange(signals.shape[1])

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = {0: "#1f77b4", 1: "#d62728"}

    for cls in (0, 1):
        cls_mask = y_te == cls
        if not np.any(cls_mask):
            continue
        cls_sig = signals[cls_mask]
        mean = cls_sig.mean(axis=0)
        sem = cls_sig.std(axis=0, ddof=0) / max(np.sqrt(cls_sig.shape[0]), 1.0)
        ax.plot(time, mean, label=f"class {cls} (n={int(cls_mask.sum())})", color=colors[cls], linewidth=1.5)
        ax.fill_between(time, mean - sem, mean + sem, color=colors[cls], alpha=0.25)

    ax.set_title(f"Shaded Error Bar (Test) - {model_name}")
    ax.set_xlabel("Time index")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig
