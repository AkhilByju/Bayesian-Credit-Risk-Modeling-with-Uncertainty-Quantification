from __future__ import annotations

import numpy as np


def expected_cost_for_threshold(
    y_true: np.ndarray,
    p_mean: np.ndarray,
    threshold: float,
    cost_fp: float = 1.0,
    cost_fn: float = 5.0,
) -> float:
    """
    Expected cost using point probabilities p_mean under a threshold decision rule.

    Predict positive (default) if p_mean >= threshold.

    Expected confusion contributions per sample:
      E[FP] = 1(yhat=1) * (1 - p)
      E[FN] = 1(yhat=0) * p

    We can also compute realized FP/FN vs y_true, but expected cost is more principled.
    """
    p = np.asarray(p_mean).astype(float)
    yhat = (p >= threshold).astype(int)

    exp_fp = (yhat == 1) * (1.0 - p)
    exp_fn = (yhat == 0) * p

    return float(cost_fp * exp_fp.mean() + cost_fn * exp_fn.mean())


def sweep_thresholds(
    y_true: np.ndarray,
    p_mean: np.ndarray,
    cost_fp: float = 1.0,
    cost_fn: float = 5.0,
    n: int = 201,
) -> dict:
    """
    Sweep thresholds from 0..1 and return arrays + best threshold.
    """
    thresholds = np.linspace(0.0, 1.0, n)
    costs = np.array(
        [expected_cost_for_threshold(y_true, p_mean, t, cost_fp, cost_fn) for t in thresholds],
        dtype=float,
    )
    best_idx = int(np.argmin(costs))
    return {
        "thresholds": thresholds,
        "costs": costs,
        "best_threshold": float(thresholds[best_idx]),
        "best_cost": float(costs[best_idx]),
    }


def uncertainty_abstain_evaluate(
    y_true: np.ndarray,
    p_mean: np.ndarray,
    p_std: np.ndarray,
    threshold: float,
    cost_fp: float = 1.0,
    cost_fn: float = 5.0,
    abstain_fracs=(0.0, 0.05, 0.1, 0.2, 0.3, 0.4),
) -> dict:
    """
    Abstain on the most uncertain fraction (largest p_std), then compute:
      - coverage (fraction kept)
      - accuracy on kept
      - expected cost on kept (using p_mean)
    """
    y_true = np.asarray(y_true).astype(int)
    p_mean = np.asarray(p_mean).astype(float)
    p_std = np.asarray(p_std).astype(float)

    idx_desc = np.argsort(-p_std)
    n = len(y_true)

    rows = []
    for frac in abstain_fracs:
        drop = int(round(frac * n))
        keep = np.ones(n, dtype=bool)
        if drop > 0:
            keep[idx_desc[:drop]] = False

        yk = y_true[keep]
        pk = p_mean[keep]

        yhat = (pk >= threshold).astype(int)
        acc = float((yhat == yk).mean())

        cost = expected_cost_for_threshold(yk, pk, threshold, cost_fp, cost_fn)

        rows.append(
            {
                "abstain_frac": float(frac),
                "coverage": float(keep.mean()),
                "accuracy": acc,
                "expected_cost": float(cost),
            }
        )

    return {"rows": rows}
