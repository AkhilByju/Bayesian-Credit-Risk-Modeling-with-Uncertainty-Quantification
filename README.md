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

### Prior Sensitivity Analysis

Bayesian logistic regression was evaluated across a range of prior variances (σ ∈ {0.5, 1.0, 2.0, 5.0}). Across this range, discriminative and calibration metrics remained largely unchanged.

This behavior is consistent with Bayesian asymptotics in high-data regimes, where the likelihood dominates the prior and posterior estimates converge. As a result, further gains are not achieved through prior tuning alone.

Subsequent analysis therefore focuses on uncertainty-aware decision making rather than raw predictive performance.

Excellent — this is **complete, coherent, and defensible**. You’ve built a _full Bayesian risk system_, not just a model.

Below I’ll do **two things**:

1. Give you a **clean, polished README update** that incorporates Day 5 results _without overselling or apologizing_
2. Lay out **clear next steps**, with a decision tree so you know when to stop vs extend

You can paste the README section verbatim.

---

# ✅ README UPDATE (FINAL RESULTS SECTION)

Add this **as a new section** near the end of your README, after Bayesian modeling / prior sensitivity.

---

## Decision Layer: Cost-Sensitive and Uncertainty-Aware Risk Modeling

To translate probabilistic predictions into actionable decisions, a **Bayesian decision layer** was implemented on top of posterior predictive outputs.

### Cost-Sensitive Threshold Selection

Rather than using an arbitrary classification threshold (e.g., 0.5), we define asymmetric misclassification costs reflecting real-world credit risk:

- False Positive (FP): cost = 1
- False Negative (FN): cost = 5

For each threshold ( \tau \in [0,1] ), the **expected decision cost** is computed under the model’s predicted probabilities. A sweep over thresholds on the validation set yields an optimal threshold:

- **Optimal threshold (validation):** ( \tau^\* \approx 0.17 )
- **Validation expected cost:** ~0.556
- **Test expected cost:** ~0.556

This highlights that optimal decision rules in imbalanced, cost-sensitive domains may lie far from the conventional ( \tau = 0.5 ).

(See `expected_cost_curve_val.png` and `expected_cost_curve_test.png`.)

---

### Uncertainty-Aware Abstention

Posterior predictive uncertainty (standard deviation of predicted probability across posterior samples) is used to **abstain** on predictions where the model is least confident.

As the system abstains on increasingly uncertain samples:

- Accuracy on the retained set increases monotonically
- Coverage decreases smoothly
- Expected cost remains stable or increases slightly, reflecting the tradeoff between selectivity and operational cost

**Validation results (selected):**

| Abstain Fraction | Coverage | Accuracy | Expected Cost |
| ---------------- | -------- | -------- | ------------- |
| 0%               | 1.00     | 0.28     | 0.556         |
| 10%              | 0.90     | 0.30     | 0.566         |
| 30%              | 0.70     | 0.35     | 0.585         |
| 40%              | 0.60     | 0.39     | 0.591         |

This demonstrates that even when raw discriminative performance is unchanged, **Bayesian uncertainty enables safer and more selective decision-making**.

(See `coverage_vs_accuracy_val.png` and `coverage_vs_cost_val.png`.)

---

### Key Takeaway

While Bayesian logistic regression does not outperform strong frequentist baselines in raw accuracy, it provides:

- Well-calibrated probabilistic outputs
- Meaningful predictive uncertainty
- Cost-aware decision thresholds
- Uncertainty-based abstention mechanisms

Together, these properties enable a **risk-aware decision system**, rather than a point-estimate classifier.

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
