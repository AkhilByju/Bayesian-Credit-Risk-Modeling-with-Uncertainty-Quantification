from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from credit_risk.viz.plots import ensure_dir


def plot_uncertainty_vs_error(
    y_true: np.ndarray,
    p_mean: np.ndarray,
    p_std: np.ndarray,
    outpath: str | Path,
    title: str = "Uncertainty vs Error (Val)",
    n_bins: int = 10,
):
    """
    Bin by uncertainty (p_std). For each bin, compute mean uncertainty and error rate.
    """
    y_true = np.asarray(y_true).astype(int)
    p_mean = np.asarray(p_mean).astype(float)
    p_std = np.asarray(p_std).astype(float)

    # predicted label at 0.5 for error rate
    y_pred = (p_mean >= 0.5).astype(int)
    err = (y_pred != y_true).astype(float)

    # bin by uncertainty quantiles
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(p_std, qs)
    # make edges strictly increasing
    edges = np.unique(edges)
    if len(edges) < 3:
        # too few unique uncertainties
        edges = np.array([p_std.min(), p_std.mean(), p_std.max()])

    bin_centers = []
    bin_err = []

    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (p_std >= lo) & (p_std <= hi) if i == len(edges) - 2 else (p_std >= lo) & (p_std < hi)
        if mask.sum() == 0:
            continue
        bin_centers.append(float(p_std[mask].mean()))
        bin_err.append(float(err[mask].mean()))

    ensure_dir(Path(outpath).parent)

    plt.figure()
    plt.plot(bin_centers, bin_err, marker="o")
    plt.xlabel("Mean posterior std of p (uncertainty)")
    plt.ylabel("Error rate (0/1 loss @ 0.5)")
    plt.title(title)
    plt.savefig(outpath, bbox_inches="tight", dpi=200)
    plt.close()


def plot_abstention_curve(
    y_true: np.ndarray,
    p_mean: np.ndarray,
    p_std: np.ndarray,
    outpath: str | Path,
    title: str = "Abstention Curve (Val)",
    fracs=(0.0, 0.05, 0.1, 0.2, 0.3, 0.4),
):
    """
    Drop the most-uncertain fraction of samples and report accuracy on the remaining.
    """
    y_true = np.asarray(y_true).astype(int)
    p_mean = np.asarray(p_mean).astype(float)
    p_std = np.asarray(p_std).astype(float)

    idx = np.argsort(-p_std)
    n = len(y_true)

    kept_fracs = []
    accs = []

    for frac in fracs:
        drop = int(round(frac * n))
        keep_mask = np.ones(n, dtype=bool)
        if drop > 0:
            keep_mask[idx[:drop]] = False

        y_pred = (p_mean[keep_mask] >= 0.5).astype(int)
        acc = (y_pred == y_true[keep_mask]).mean()

        kept_fracs.append(1.0 - frac)
        accs.append(float(acc))

    ensure_dir(Path(outpath).parent)

    plt.figure()
    plt.plot(kept_fracs, accs, marker="o")
    plt.xlabel("Fraction kept (1 - abstention)")
    plt.ylabel("Accuracy on kept set")
    plt.title(title)
    plt.savefig(outpath, bbox_inches="tight", dpi=200)
    plt.close()
