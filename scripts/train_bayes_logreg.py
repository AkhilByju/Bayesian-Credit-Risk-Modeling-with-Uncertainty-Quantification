from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

from credit_risk.eval.metrics import compute_classification_metrics
from credit_risk.viz.plots import plot_roc, plot_pr, plot_calibration, ensure_dir
from credit_risk.models.bayes_logreg import build_bayes_logreg_model

def load_split(path: Path):
    df = pd.read_csv(path)
    X = df.drop(columns=["y"])
    y = df["y"].astype(int)
    return X, y

def posterior_predict_proba(idata) -> tuple[np.ndarray, np.ndarray]:
    """
    Finds p draws in whichever group PyMC used:
      - idata.predictions["p"]          (common with predictions=True)
      - idata.posterior_predictive["p"]
      - idata.posterior["p"]
    Returns mean/std across draws per observation.
    """
    p = None

    if hasattr(idata, "predictions") and idata.predictions is not None and "p" in idata.predictions:
        p = idata.predictions["p"].values
    elif hasattr(idata, "posterior_predictive") and idata.posterior_predictive is not None and "p" in idata.posterior_predictive:
        p = idata.posterior_predictive["p"].values
    elif hasattr(idata, "posterior") and idata.posterior is not None and "p" in idata.posterior:
        p = idata.posterior["p"].values
    elif isinstance(idata, dict) and "p" in idata:
        p = np.asarray(idata["p"])

    if p is None:
        groups = []
        try:
            groups = list(getattr(idata, "groups", lambda: [])())
        except Exception:
            pass
        raise AttributeError(f"Could not find 'p'. Available groups: {groups}")

    # p expected shape: (chain, draw, obs) OR (draw, obs)
    if p.ndim == 3:
        p_flat = p.reshape(-1, p.shape[-1])
    elif p.ndim == 2:
        p_flat = p
    else:
        raise ValueError(f"Unexpected shape for p: {p.shape}")

    return p_flat.mean(axis=0), p_flat.std(axis=0)

def proba_from_posterior_params(idata: az.InferenceData, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    alpha = idata.posterior["alpha"].values  # (chain, draw)
    beta = idata.posterior["beta"].values    # (chain, draw, feature)

    alpha_f = alpha.reshape(-1)              # (S,)
    beta_f = beta.reshape(-1, beta.shape[-1])# (S, F)

    Xv = X.to_numpy(dtype=float)             # (N, F)
    logits = alpha_f[:, None] + beta_f @ Xv.T  # (S, N)
    p = 1.0 / (1.0 + np.exp(-logits))        # (S, N)

    return p.mean(axis=0), p.std(axis=0)

def main():
    seed = 42
    processed = Path("data/processed")
    figs = ensure_dir("reports/figures")
    out_dir = ensure_dir("reports/posterior")
    out_metrics = Path("reports/metrics_bayes_logreg.json")

    X_train, y_train = load_split(processed / "train.csv")
    X_val, y_val = load_split(processed / "val.csv")
    X_test, y_test = load_split(processed / "test.csv")

    # build model
    model = build_bayes_logreg_model(X_train, y_train, prior_sigma=1.0)

    results = {
        "config": {"seed": seed, "prior_sigma": 1.0}
    }

    with model:
        map_est = pm.find_MAP()
        results["map"] = {"alpha": float(map_est["alpha"])}

        idata = pm.sample(
            draws=800,
            tune=800,
            chains=2,
            cores=1,               
            target_accept=0.9,
            random_seed=seed,
            return_inferencedata=True,
            progressbar=True,
        )

        # save everything
        az.to_netcdf(idata, out_dir / "bayes_logreg_mcmc.nc")

        summary = az.summary(idata, var_names=["alpha", "beta"], kind="stats").reset_index()
        summary.to_csv(out_dir / "bayes_logreg_summary.csv", index=False)

        # Prediction
        pm.set_data({"X": X_val.to_numpy(dtype=float), "y": y_val.to_numpy(dtype=int)})
        idata_val = pm.sample_posterior_predictive(
                idata,
                var_names=["p"],
                random_seed=seed,
                predictions=True,          
                return_inferencedata=True
            )
        mean_val_p, std_val_p = proba_from_posterior_params(idata, X_val)

        pm.set_data({"X": X_test.to_numpy(dtype=float), "y": y_test.to_numpy(dtype=int)})
        idata_test = pm.sample_posterior_predictive(
                idata,
                var_names=["p"],
                random_seed=seed,
                predictions=True,          
                return_inferencedata=True
            )
        mean_test_p, std_test_p = proba_from_posterior_params(idata, X_test)

    # Metrics
    results["mcmc"] = {
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

    # Plots
    plot_roc(y_val, mean_val_p, figs / "bayes_logreg_roc_val.png", "Bayesian LogReg ROC (Val)")
    plot_pr(y_val, mean_val_p, figs / "bayes_logreg_pr_val.png", "Bayesian LogReg PR (Val)")
    plot_calibration(y_val, mean_val_p, figs / "bayes_logreg_cal_val.png", "Bayesian LogReg Calibration (Val)")


    out_metrics.write_text(json.dumps(results, indent=2))
    print(f"✅ Wrote: {out_metrics}")
    print(f"✅ Posterior saved: {out_dir / 'bayes_logreg_mcmc.nc'}")
    print("✅ Plots in: reports/figures/")
    print(json.dumps(results["mcmc"], indent=2))

if __name__ == "__main__":
    main()
