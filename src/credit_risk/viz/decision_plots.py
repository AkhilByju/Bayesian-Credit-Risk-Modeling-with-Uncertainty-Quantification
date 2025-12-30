from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from credit_risk.viz.plots import ensure_dir


def plot_cost_curve(thresholds: np.ndarray, costs: np.ndarray, outpath: str | Path, title: str):
    ensure_dir(Path(outpath).parent)
    plt.figure()
    plt.plot(thresholds, costs)
    plt.xlabel("Decision threshold τ")
    plt.ylabel("Expected cost")
    plt.title(title)
    plt.savefig(outpath, bbox_inches="tight", dpi=200)
    plt.close()


def plot_coverage_curve(rows: list[dict], outpath: str | Path, title: str, y_key: str):
    ensure_dir(Path(outpath).parent)
    cov = [r["coverage"] for r in rows]
    yv = [r[y_key] for r in rows]
    plt.figure()
    plt.plot(cov, yv, marker="o")
    plt.xlabel("Coverage (fraction kept)")
    plt.ylabel(y_key.replace("_", " ").title())
    plt.title(title)
    plt.savefig(outpath, bbox_inches="tight", dpi=200)
    plt.close()
