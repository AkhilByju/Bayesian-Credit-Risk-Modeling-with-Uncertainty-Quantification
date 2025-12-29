from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from sklearn.calibration import CalibrationDisplay

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def plot_roc(y_true, y_proba, outpath: str | Path, title: str):
    plt.figure()
    RocCurveDisplay.from_predictions(y_true, y_proba)
    plt.title(title)
    plt.savefig(outpath, bbox_inches="tight", dpi=200)
    plt.close()

def plot_pr(y_true, y_proba, outpath: str | Path, title: str):
    plt.figure()
    PrecisionRecallDisplay.from_predictions(y_true, y_proba)
    plt.title(title)
    plt.savefig(outpath, bbox_inches="tight", dpi=200)
    plt.close()


def plot_calibration(y_true, y_proba, outpath: str | Path, title: str, n_bins: int = 10):
    plt.figure()
    CalibrationDisplay.from_predictions(y_true, y_proba, n_bins=n_bins)
    plt.title(title)
    plt.savefig(outpath, bbox_inches="tight", dpi=200)
    plt.close()