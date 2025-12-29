from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier


def train_logreg(X, y, seed: int = 42) -> LogisticRegression:
    # Starter Parameters
    model = LogisticRegression(
        max_iter=2000,
        solver='lbfgs',
        class_weight="balanced",
        random_state=seed,
        n_jobs=None,
    )
    model.fit(X, y)
    return model

def train_xgb(X, y, seed: int = 42):
    # Starter Parameters
    model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=4,
    )
    model.fit(X, y)
    return model

def predict_proba(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]
