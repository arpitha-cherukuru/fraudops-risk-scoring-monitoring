# FraudOps Risk Scoring & Monitoring  
## End-to-End UK Payments Fraud Simulation (1.2M Transactions)

This project simulates how a UK fintech fraud team would design, deploy, optimise and monitor a real-world fraud decisioning system under business and operational constraints.

Most fraud ML projects stop at model accuracy.  
This project models cost trade-offs, review capacity limits, SLA pressure, monitoring, retraining triggers, and explainability to reflect how fraud systems behave in production environments.

---

## What This Project Demonstrates

- End-to-end fraud risk scoring pipeline  
- Cost-based threshold optimisation (not accuracy-based)  
- Manual review queue simulation under capacity limits  
- SLA breach monitoring and backlog build-up  
- Operational KPI generation  
- Model drift detection using PR-AUC tracking  
- Explainability for governance and compliance  
- Executive-ready Power BI dashboards  

This is a full Fraud Operations simulation — not just a machine learning model.

---

## Architecture Overview

Synthetic Transaction Generation  
→ Time-Based Train/Validation/Test Split  
→ Fraud Probability Model (Logistic Regression)  
→ Cost-Based Decision Engine  
→ Decision Application & KPI Generation  
→ Manual Review Queue Simulation  
→ Drift Monitoring  
→ Executive Dashboards (Power BI)

---

## Project Overview

A realistic synthetic UK payments dataset was generated containing **1.2 million transactions** with fraud-like behavioural signals.

### Behavioural Signals Simulated

- Transaction amount (GBP)  
- Account age  
- IP risk score  
- Distance from home  
- Transaction velocity (last hour)  
- Night / weekend activity  
- Chargeback history  
- Merchant category  
- Channel (card-present, online, etc.)

The data is synthetic (no real banking data) but structured to behave like real fraud analytics datasets.

---

## Pipeline Steps

### 1. Dataset Generation  
`01_make_dataset.py`  
Creates 1.2M synthetic transactions with fraud labels.  

Output:  
`data/raw/uk_payments_synthetic_1_2m.csv`

---

### 2. Time-Based Train / Validation / Test Split  
`02_preprocess.py`  
Splits data chronologically to prevent data leakage and mimic production deployment.

---

### 3. Fraud Risk Model  
`03_train_model.py`  
Trains a Logistic Regression model and outputs fraud probability scores.

---

### 4. Cost-Based Decision Engine  
`04_decision_engine.py`  

Optimises:

- Review threshold  
- Decline threshold  

Objective:

Total Cost = Missed Fraud Loss + Review Cost + False Decline Cost

This reflects real fraud strategy trade-offs between customer experience and fraud prevention.

---

### 5. Apply Decisions & KPI Generation  
`05_apply_decisions.py`

Generates:

- `decisions.csv`  
- `daily_kpis.csv`  

These outputs power executive and operational dashboards.

---

### 6. Manual Review Queue Simulation  
`06_case_queue.py`

Simulates:

- Daily review capacity  
- SLA limits  
- Backlog build-up  
- SLA breaches  

Models operational stress when investigation capacity is constrained.

---

### 7. Drift Monitoring  
`07_drift_monitor.py`

Tracks weekly PR-AUC performance and flags retraining when degradation is detected.

Simulates basic model governance in production.

---

### 8. Model Explainability  
`08_model_explainability.py`

Extracts feature importance from the trained model to support transparency and compliance requirements.

---

## Key Results

- 1.2M transactions simulated  
- 47,051 transactions escalated to manual review  
- 41,876 SLA breaches under constrained review capacity  
- Cost-based threshold optimisation reduced total fraud operational cost  
- Drift monitoring flagged 2 degradation weeks requiring retraining  
- Built 3 production-style Power BI dashboards  

---

## Power BI Dashboards

Three dashboards simulate what different fraud stakeholders would use daily:

1. Executive Fraud Overview  
   - Fraud rate trends  
   - Missed fraud  
   - False declines  
   - Total fraud operational cost  

2. Decision & Cost Analysis  
   - Threshold impact  
   - Cost breakdown  
   - Fraud vs business trade-offs  

3. Investigator Operations Dashboard  
   - Review queue backlog  
   - SLA breaches  
   - Average backlog days  
   - Risk drivers in review queue  

Dashboard screenshots are available in the `assets/` folder.

---

## Tech Stack

- Python  
- Pandas & NumPy  
- Scikit-learn (Logistic Regression)  
- Power BI  
- Time-based validation methodology  
- Cost optimisation logic  
- PR-AUC based drift monitoring  
- Git & GitHub  

---

## How to Run

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Full Pipeline

From the project root directory:

```bash
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

Real fraud systems are not just models.

They operate under:

- Business cost trade-offs  
- Investigation team constraints  
- SLA obligations  
- Regulatory governance requirements  
- Continuous monitoring needs  

This project demonstrates how fraud systems behave under operational pressure — bridging the gap between machine learning, risk strategy, and fraud operations.
