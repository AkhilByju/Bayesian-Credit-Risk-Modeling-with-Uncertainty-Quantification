from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm

def build_bayes_logreg_model(X: pd.DataFrame, y: pd.Series, prior_sigma: float = 1.0) -> pm.Model:
    Xv = X.to_numpy(dtype=float)
    yv = y.to_numpy(dtype=int)

    coords = {
        "feature": np.array(X.columns),
    }

    with pm.Model(coords=coords) as model:
        X_data = pm.Data("X", Xv, dims=("obs", "feature"))
        y_data = pm.Data("y", yv, dims=("obs",))

        alpha = pm.Normal("alpha", mu=0.0, sigma=5.0)
        beta = pm.Normal("beta", mu=0.0, sigma=prior_sigma, dims=("feature",))

        logits = alpha + pm.math.dot(X_data, beta)
        p = pm.Deterministic("p", pm.math.sigmoid(logits), dims=("obs",))

        pm.Bernoulli("likelihood", p=p, observed=y_data, dims=("obs",))
        
    return model