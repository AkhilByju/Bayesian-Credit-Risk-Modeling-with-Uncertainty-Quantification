from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

from credit_risk.models.bayes_logreg import build_bayes_logreg_model
from credit_risk.eval.metrics import compute_classification_metrics
from credit_risk.viz.plots import ensure_dir, plot_roc, plot_pr, plot_calibration
from credit_risk.viz.uncertainty_plots import plot_uncertainty_vs_error, plot_abstention_curve


def load_split(path: Path):
    df = pd.read_csv(path)
    X = df.drop(columns=["y"])
    y = df["y"].astype(int).to_numpy()
    return X, y


def proba_from_posterior_params(idata: az.InferenceData, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute p = sigmoid(alpha + X @ beta) for each posterior draw.
    Returns mean/std over draws per sample.
    """
    alpha = idata.posterior["alpha"].values  # (chain, draw)
    beta = idata.posterior["beta"].values    # (chain, draw, feature)

    alpha_f = alpha.reshape(-1)                     # (S,)
    beta_f = beta.reshape(-1, beta.shape[-1])       # (S, F)

    Xv = X.to_numpy(dtype=float)                    # (N, F)
    logits = alpha_f[:, None] + beta_f @ Xv.T       # (S, N)
    p = 1.0 / (1.0 + np.exp(-logits))               # (S, N)

    return p.mean(axis=0), p.std(axis=0)


def main():
    seed = 42
    processed = Path("data/processed")
    figs = ensure_dir("reports/figures")
    out_dir = ensure_dir("reports/posterior")
    out_metrics = Path("reports/metrics_bayes_prior_sweep.json")

    X_train, y_train = load_split(processed / "train.csv")
    X_val, y_val = load_split(processed / "val.csv")
    X_test, y_test = load_split(processed / "test.csv")

    # Prior sweep (you can add 10.0 later if needed)
    priors = [0.5, 1.0, 2.0, 5.0]

    results = {"config": {"seed": seed, "priors": priors}, "runs": {}}

    for prior_sigma in priors:
        key = f"sigma_{prior_sigma}"
        print(f"\n=== Running Bayesian LogReg: prior_sigma={prior_sigma} ===")

        model = build_bayes_logreg_model(X_train, pd.Series(y_train), prior_sigma=prior_sigma)

        with model:
            idata = pm.sample(
                draws=600,
                tune=600,
                chains=2,
                cores=1,              # important for mac stability
                target_accept=0.9,
                random_seed=seed,
                return_inferencedata=True,
                progressbar=True,
            )

        # Save posterior
        az.to_netcdf(idata, out_dir / f"bayes_logreg_{key}.nc")

        # Predict mean/std on val/test
        mean_val_p, std_val_p = proba_from_posterior_params(idata, X_val)
        mean_test_p, std_test_p = proba_from_posterior_params(idata, X_test)

        run = {
            "val": compute_classification_metrics(y_val, mean_val_p),
            "test": compute_classification_metrics(y_test, mean_test_p),
            "val_uncertainty": {
                "mean_std_p": float(std_val_p.mean()),
                "p95_std_p": float(np.quantile(std_val_p, 0.95)),
            },
            "test_uncertainty": {
                "mean_std_p": float(std_test_p.mean()),
                "p95_std_p": float(np.quantile(std_test_p, 0.95)),
            },
        }
        results["runs"][key] = run

        # Plots for val
        plot_roc(y_val, mean_val_p, figs / f"bayes_{key}_roc_val.png", f"Bayes LogReg ROC (Val) σ={prior_sigma}")
        plot_pr(y_val, mean_val_p, figs / f"bayes_{key}_pr_val.png", f"Bayes LogReg PR (Val) σ={prior_sigma}")
        plot_calibration(y_val, mean_val_p, figs / f"bayes_{key}_cal_val.png", f"Bayes LogReg Calibration (Val) σ={prior_sigma}")

        # Extra “Bayesian” plots (only once for the best sigma later; for now do it for all or pick one)
        plot_uncertainty_vs_error(
            y_val, mean_val_p, std_val_p,
            figs / f"bayes_{key}_uncertainty_vs_error_val.png",
            title=f"Uncertainty vs Error (Val) σ={prior_sigma}",
        )
        plot_abstention_curve(
            y_val, mean_val_p, std_val_p,
            figs / f"bayes_{key}_abstention_val.png",
            title=f"Abstention Curve (Val) σ={prior_sigma}",
            fracs=(0.0, 0.05, 0.1, 0.2, 0.3, 0.4),
        )

    out_metrics.write_text(json.dumps(results, indent=2))
    print(f"\n✅ Wrote: {out_metrics}")

    # Create an easy CSV summary for README
    rows = []
    for k, v in results["runs"].items():
        rows.append({
            "prior": k,
            "val_roc_auc": v["val"]["roc_auc"],
            "val_pr_auc": v["val"]["pr_auc"],
            "val_brier": v["val"]["brier"],
            "val_log_loss": v["val"]["log_loss"],
            "val_acc": v["val"]["accuracy"],
            "val_mean_std_p": v["val_uncertainty"]["mean_std_p"],
            "val_p95_std_p": v["val_uncertainty"]["p95_std_p"],
        })
    df = pd.DataFrame(rows).sort_values("val_roc_auc", ascending=False)
    csv_path = Path("reports/figures/prior_sweep_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Wrote: {csv_path}")


if __name__ == "__main__":
    main()
