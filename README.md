# Bayesian Credit Risk Modeling with Uncertainty Quantification

## Overview

This project implements a **probabilistic credit risk prediction system** by modeling **uncertainty** in predictions. This project investigates both **frequentist** and **Bayesian** approaches, with an emphasis on **calibration, uncertainty estimation, and decision-aware modeling**.

---

## Dataset

**UCI Default of Credit Card Clients Dataset**

- 30,000 credit card holders
- 23 original features including credit limit, payment history, bill amounts, and demographic indicators
- Binary target: default in the following month

The dataset is moderately imbalanced and noisy, making it a realistic benchmark for risk modeling.

---

## Methodology

### Data Preprocessing

- Removed metadata rows and ID columns
- Stratified train/validation/test split:

  - Train: 21,675
  - Validation: 3,825
  - Test: 4,500

- Numeric features standardized
- Categorical features one-hot encoded
- All preprocessing fit **only on training data** to avoid leakage

Final feature dimensionality after encoding: **91 features**

---

## Baseline Models (Frequentist)

Two baseline models were implemented to establish strong non-Bayesian performance benchmarks:

### Logistic Regression

- L2-regularized
- Class-weighted to handle imbalance
- Interpretable linear decision boundary

### XGBoost

- Gradient-boosted decision trees
- Captures nonlinear feature interactions
- Strong empirical performance but limited uncertainty modeling

---

## Results

### Validation Metrics

| Model               | ROC-AUC | PR-AUC | Log Loss | Brier | Accuracy |
| ------------------- | ------- | ------ | -------- | ----- | -------- |
| Logistic Regression | 0.778   | 0.556  | 0.555    | 0.181 | 0.782    |
| XGBoost             | 0.784   | 0.566  | 0.426    | 0.132 | 0.822    |

### Test Metrics

| Model               | ROC-AUC | PR-AUC | Log Loss | Brier | Accuracy |
| ------------------- | ------- | ------ | -------- | ----- | -------- |
| Logistic Regression | 0.764   | 0.528  | 0.562    | 0.184 | 0.776    |
| XGBoost             | 0.779   | 0.558  | 0.430    | 0.135 | 0.820    |

### Bayesian Logistic Regression (Initial Results)

A Bayesian logistic regression model with Gaussian priors on coefficients was implemented and trained using full posterior sampling (NUTS).

With a relatively strong prior (σ = 1.0), the model exhibited:

- Lower discriminative performance compared to frequentist baselines
- Significantly higher posterior predictive uncertainty

This behavior is consistent with strong regularization imposed by the prior, which shrinks coefficients toward zero and favors conservative predictions. The results highlight the sensitivity of Bayesian models to prior assumptions and motivate a systematic prior sensitivity analysis.

Uncertainty statistics (validation):

- Mean posterior std of predicted probability: ~0.22
- 95th percentile posterior std: ~0.44

---

## Calibration Analysis

Despite strong discrimination performance, both models produce **miscalibrated probability estimates**, especially under class imbalance. XGBoost improves predictive accuracy but still outputs point estimates without reliable uncertainty bounds.

This motivates the use of **Bayesian probabilistic models**, which will be introduced next to:

- Quantify posterior uncertainty
- Improve calibration
- Enable risk-aware decision thresholds

(See `reports/figures/` for ROC, PR, and calibration plots.)

---

## Next Steps

Planned extensions:

- Bayesian logistic regression with prior modeling
- MAP vs full Bayesian inference (MCMC)
- Variational inference (ADVI)
- Posterior predictive uncertainty analysis
- Cost-sensitive decision thresholds using Bayesian decision theory

---

## Reproducibility

```bash
# Preprocess data
python scripts/preprocess.py

# Train baseline models
python scripts/train_baselines.py
```

All experiments use fixed random seeds and deterministic preprocessing.

---

## Project Structure

```
bayes-credit-risk/
├── data/
│   ├── raw/
│   ├── processed/
├── src/credit_risk/
│   ├── data.py
│   ├── features.py
│   ├── models/
│   ├── eval/
│   ├── viz/
├── scripts/
├── reports/
│   ├── figures/
└── README.md
```

---

## Key Takeaway

Strong discriminative performance does not guarantee reliable probability estimates. This project demonstrates why **uncertainty-aware Bayesian modeling** is critical for real-world risk systems.

---
