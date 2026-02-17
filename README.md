# FraudOps Risk Scoring & Monitoring (UK Payments Simulation)

This project simulates how a real UK fintech fraud team would design, deploy and monitor a fraud decisioning system.

It is not just a machine learning model.
It is a full fraud operations simulation including:

- Risk scoring
- Threshold optimisation
- Manual review queue simulation
- SLA breach monitoring
- Cost analysis
- Drift detection
- Model explainability
- Power BI executive dashboards

---

## Project Overview

I generated a realistic synthetic UK payments dataset with 1.2 million transactions.

The dataset includes fraud-like behavioural signals such as:

- Transaction amount (GBP)
- Account age
- IP risk score
- Distance from home
- Velocity in last hour
- Night / weekend behaviour
- Chargeback history
- Merchant category
- Channel

The data is synthetic (not real banking data) but structured to behave like real fraud analytics data.

---

## Pipeline Steps

### 1. Dataset Generation
`01_make_dataset.py`

Creates 1.2M synthetic transactions with fraud labels.

Output:
data/raw/uk_payments_synthetic_1_2m.csv

---

### 2. Time-Based Train/Validation/Test Split
`02_preprocess.py`

Splits data chronologically to avoid data leakage.

---

### 3. Fraud Risk Model
`03_train_model.py`

Trains a Logistic Regression model.
Outputs fraud probability scores.

---

### 4. Cost-Based Decision Engine
`04_decision_engine.py`

Optimises:
- Review threshold
- Decline threshold

Goal:
Minimise total operational cost (missed fraud + review cost + false declines).

---

### 5. Apply Decisions + KPI Generation
`05_apply_decisions.py`

Generates:
- decisions.csv
- daily_kpis.csv

These power the Power BI dashboards.

---

### 6. Manual Review Queue Simulation
`06_case_queue.py`

Simulates:
- Daily review capacity
- SLA limits
- Backlog build-up
- SLA breaches

This models operational stress when review capacity is limited.

---

### 7. Drift Monitoring
`07_drift_monitor.py`

Tracks weekly PR-AUC.
Flags retraining when performance drops.

---

### 8. Model Explainability
`08_model_explainability.py`

Extracts feature importance from the logistic regression model.

---

## Key Results

- 1.2M transactions simulated
- 47K transactions sent to review
- Review capacity stress caused SLA breaches
- Cost-based threshold optimisation reduced total fraud cost
- Drift monitoring flagged performance degradation weeks

---

## Power BI Dashboards

Three dashboards were built:

1. Executive Fraud Overview
2. Decision & Cost Analysis
3. Investigator Operations

These dashboards simulate what a fraud strategy team would use daily.

---

## How to Run

From project root:

```
python src/01_make_dataset.py
python src/02_preprocess.py
python src/03_train_model.py
python src/04_decision_engine.py
python src/05_apply_decisions.py
python src/06_case_queue.py
python src/07_drift_monitor.py
python src/08_model_explainability.py
```

---

## Why This Project Matters

Most fraud ML projects stop at model accuracy.

This project demonstrates:

- Business cost trade-offs
- Operational constraints
- Review team capacity modelling
- SLA impact
- Governance and explainability
- Monitoring and retraining triggers

It shows how fraud systems behave in real-world fintech environments.
