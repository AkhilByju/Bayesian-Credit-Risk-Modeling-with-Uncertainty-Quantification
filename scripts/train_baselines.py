from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from credit_risk.models.baseline import train_logreg, train_xgb, predict_proba
from credit_risk.eval.metrics import compute_classification_metrics
from credit_risk.viz.plots import plot_roc, plot_pr, plot_calibration, ensure_dir

def load_split(path: Path):
    df = pd.read_csv(path)
    X = df.drop(columns=["y"])
    y = df["y"].astype(int)
    return X, y

def main():
    seed = 42
    processed = Path("data/processed")
    figs = ensure_dir("reports/figures")
    out_metrics = Path("reports.metrics_baselines.json")

    X_train, y_train = load_split(processed / "train.csv")
    X_val, y_val = load_split(processed / "val.csv")
    X_test, y_test = load_split(processed / "test.csv")

    results = {}

    # Logistic Regression
    logreg = train_logreg(X_train, y_train, seed=seed)
    val_p = predict_proba(logreg, X_val)
    test_p = predict_proba(logreg, X_test)

    # store results
    results["logreg"] = {
        "val": compute_classification_metrics(y_val, val_p),
        "test": compute_classification_metrics(y_test, test_p),
    }

    # plot graphs
    plot_roc(y_val, val_p, figs / "logreg_roc_val.png", "LogReg ROC (Val)")
    plot_pr(y_val, val_p, figs / "logreg_pr_val.png", "LogReg PR (Val)")
    plot_calibration(y_val, val_p, figs / "logreg_cal_val.png", "LogReg Calibration (Val)")

    # XGBoost 
    try:
        xgb = train_xgb(X_train, y_train, seed=seed)
        val_p = predict_proba(xgb, X_val)
        test_p = predict_proba(xgb, X_test)

        # store results
        results["xgb"] = {
            "val": compute_classification_metrics(y_val, val_p),
            "test": compute_classification_metrics(y_test, test_p),
        }
        
        # plot graphs
        plot_roc(y_val, val_p, figs / "xgb_roc_val.png", "XGBoost ROC (Val)")
        plot_pr(y_val, val_p, figs / "xgb_pr_val.png", "XGBoost PR (Val)")
        plot_calibration(y_val, val_p, figs / "xgb_cal_val.png", "XGBoost Calibration (Val)")
    except Exception as e:
        results["xgb_error"] = str(e)


    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.write_text(json.dumps(results, indent=2))
    print(f"✅ Wrote metrics: {out_metrics}")
    print("✅ Wrote plots to: reports/figures/")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()