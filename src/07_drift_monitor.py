import os
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

TEST_PRED_PATH = "outputs/test_predictions.csv"
TEST_FEATURES_PATH = "data/processed/test.csv"
OUT_PATH = "outputs/drift_report.csv"

# PSI settings
PSI_BINS = 10
PSI_ALERT = 0.10  # >0.2 moderate drift

# Retrain triggers
PR_AUC_DROP_TRIGGER = 0.10  # 20% drop vs baseline


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between expected and actual distributions."""
    eps = 1e-6
    quantiles = np.linspace(0, 1, bins + 1)
    breakpoints = np.unique(np.quantile(expected, quantiles))

    # If feature has too few unique values, PSI isn't meaningful
    if len(breakpoints) < 3:
        return 0.0

    exp_counts, _ = np.histogram(expected, bins=breakpoints)
    act_counts, _ = np.histogram(actual, bins=breakpoints)

    exp_perc = exp_counts / max(exp_counts.sum(), 1)
    act_perc = act_counts / max(act_counts.sum(), 1)

    exp_perc = np.clip(exp_perc, eps, 1)
    act_perc = np.clip(act_perc, eps, 1)

    return float(np.sum((act_perc - exp_perc) * np.log(act_perc / exp_perc)))


def ensure_dir(path: str) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)


def main():
    ensure_dir(OUT_PATH)

    preds = pd.read_csv(TEST_PRED_PATH, parse_dates=["timestamp"])
    feats = pd.read_csv(TEST_FEATURES_PATH, parse_dates=["timestamp"])

    df = preds.merge(
        feats,
        on=["transaction_id", "timestamp", "fraud_label"],
        how="left"
    )

    df["week"] = df["timestamp"].dt.to_period("W").astype(str)

    drift_features = [
        "amount_gbp",
        "velocity_1h",
        "ip_risk_score",
        "account_age_days",
        "amount_vs_avg_30d"
    ]

    weeks = sorted(df["week"].unique())

    # Baseline = first 2 weeks of test
    baseline_weeks = weeks[:2]
    baseline_df = df[df["week"].isin(baseline_weeks)].copy()

    baseline_pr_auc = average_precision_score(
        baseline_df["fraud_label"], baseline_df["risk_score"]
    )

    rows = []

    for w in weeks:
        wk = df[df["week"] == w].copy()

        pr_auc = average_precision_score(wk["fraud_label"], wk["risk_score"])

        psi_vals = {f: psi(baseline_df[f].values, wk[f].values, bins=PSI_BINS) for f in drift_features}
        psi_max = max(psi_vals.values())

        pr_auc_drop_pct = (baseline_pr_auc - pr_auc) / max(baseline_pr_auc, 1e-6)

        retrain_flag = 1 if (psi_max > PSI_ALERT or pr_auc_drop_pct > PR_AUC_DROP_TRIGGER) else 0

        rows.append({
            "week": w,
            "pr_auc": float(pr_auc),
            "baseline_pr_auc": float(baseline_pr_auc),
            "pr_auc_drop_pct": float(pr_auc_drop_pct),
            "psi_amount_gbp": float(psi_vals["amount_gbp"]),
            "psi_velocity_1h": float(psi_vals["velocity_1h"]),
            "psi_ip_risk_score": float(psi_vals["ip_risk_score"]),
            "psi_account_age_days": float(psi_vals["account_age_days"]),
            "psi_amount_vs_avg_30d": float(psi_vals["amount_vs_avg_30d"]),
            "psi_max": float(psi_max),
            "retrain_flag": int(retrain_flag)
        })

    report = pd.DataFrame(rows)
    report.to_csv(OUT_PATH, index=False)

    print("Created outputs/drift_report.csv")
    print(f"Baseline PR-AUC (first 2 weeks): {baseline_pr_auc:.4f}")
    print(f"Retrain flags: {int(report['retrain_flag'].sum())} weeks flagged")


if __name__ == "__main__":
    main()
