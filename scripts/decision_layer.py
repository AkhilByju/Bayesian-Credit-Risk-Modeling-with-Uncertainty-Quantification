from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import arviz as az

from credit_risk.eval.decision import sweep_thresholds, uncertainty_abstain_evaluate
from credit_risk.viz.decision_plots import plot_cost_curve, plot_coverage_curve
from credit_risk.viz.plots import ensure_dir


def load_split(path: Path):
    df = pd.read_csv(path)
    X = df.drop(columns=["y"])
    y = df["y"].astype(int).to_numpy()
    return X, y


def proba_from_posterior_params(idata: az.InferenceData, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    alpha = idata.posterior["alpha"].values  # (chain, draw)
    beta = idata.posterior["beta"].values    # (chain, draw, feature)

    alpha_f = alpha.reshape(-1)
    beta_f = beta.reshape(-1, beta.shape[-1])

    Xv = X.to_numpy(dtype=float)
    logits = alpha_f[:, None] + beta_f @ Xv.T
    p = 1.0 / (1.0 + np.exp(-logits))

    return p.mean(axis=0), p.std(axis=0)


def main():
    # Choose which posterior file to use from Day 4
    # e.g. reports/posterior/bayes_logreg_sigma_2.0.nc
    posterior_path = Path("reports/posterior/bayes_logreg_sigma_2.0.nc")

    # Costs: tweak these to tell a story (FN usually more expensive than FP)
    cost_fp = 1.0
    cost_fn = 5.0

    processed = Path("data/processed")
    figs = ensure_dir("reports/figures")
    out_metrics = Path("reports/metrics_decision_layer.json")

    X_train, y_train = load_split(processed / "train.csv")
    X_val, y_val = load_split(processed / "val.csv")
    X_test, y_test = load_split(processed / "test.csv")

    idata = az.from_netcdf(posterior_path)

    mean_val_p, std_val_p = proba_from_posterior_params(idata, X_val)
    mean_test_p, std_test_p = proba_from_posterior_params(idata, X_test)

    # --- Threshold sweep on validation ---
    sweep_val = sweep_thresholds(y_val, mean_val_p, cost_fp=cost_fp, cost_fn=cost_fn, n=401)
    best_tau = sweep_val["best_threshold"]

    # --- Evaluate expected cost curve (val + test) ---
    sweep_test = sweep_thresholds(y_test, mean_test_p, cost_fp=cost_fp, cost_fn=cost_fn, n=401)

    plot_cost_curve(
        sweep_val["thresholds"], sweep_val["costs"],
        figs / "expected_cost_curve_val.png",
        title=f"Expected Cost vs Threshold (Val)  FP={cost_fp}, FN={cost_fn}"
    )
    plot_cost_curve(
        sweep_test["thresholds"], sweep_test["costs"],
        figs / "expected_cost_curve_test.png",
        title=f"Expected Cost vs Threshold (Test)  FP={cost_fp}, FN={cost_fn}"
    )

    # --- Abstention: drop most uncertain and evaluate on kept set using best_tau ---
    abst_val = uncertainty_abstain_evaluate(
        y_val, mean_val_p, std_val_p,
        threshold=best_tau,
        cost_fp=cost_fp,
        cost_fn=cost_fn,
        abstain_fracs=(0.0, 0.05, 0.1, 0.2, 0.3, 0.4),
    )
    abst_test = uncertainty_abstain_evaluate(
        y_test, mean_test_p, std_test_p,
        threshold=best_tau,
        cost_fp=cost_fp,
        cost_fn=cost_fn,
        abstain_fracs=(0.0, 0.05, 0.1, 0.2, 0.3, 0.4),
    )

    plot_coverage_curve(
        abst_val["rows"], figs / "coverage_vs_cost_val.png",
        title="Coverage vs Expected Cost (Val, uncertainty-based abstention)",
        y_key="expected_cost",
    )
    plot_coverage_curve(
        abst_val["rows"], figs / "coverage_vs_accuracy_val.png",
        title="Coverage vs Accuracy (Val, uncertainty-based abstention)",
        y_key="accuracy",
    )

    results = {
        "config": {
            "posterior_path": str(posterior_path),
            "cost_fp": cost_fp,
            "cost_fn": cost_fn,
        },
        "threshold_sweep": {
            "val_best_threshold": best_tau,
            "val_best_cost": sweep_val["best_cost"],
            "test_best_threshold": sweep_test["best_threshold"],
            "test_best_cost": sweep_test["best_cost"],
        },
        "abstention": {
            "val": abst_val,
            "test": abst_test,
        },
    }

    out_metrics.write_text(json.dumps(results, indent=2))
    print(f"✅ Wrote: {out_metrics}")
    print("✅ Plots in: reports/figures/")
    print(json.dumps(results["threshold_sweep"], indent=2))


if __name__ == "__main__":
    main()
