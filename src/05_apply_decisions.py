import os
import pandas as pd
import numpy as np

PRED_PATH = "outputs/test_predictions.csv"
TEST_PATH = "data/processed/test.csv"
OUT_DIR = "outputs"

REVIEW_TH = 0.10
DECLINE_TH = 0.80


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main():
    ensure_dir(OUT_DIR)

    # Load model predictions
    preds = pd.read_csv(PRED_PATH, parse_dates=["timestamp"])

    # Load test features to add drill-down columns (amount, channel, etc.)
    test_df = pd.read_csv(TEST_PATH, parse_dates=["timestamp"])

    # Merge predictions with features
    df = preds.merge(
        test_df,
        on=["transaction_id", "timestamp", "fraud_label"],
        how="left"
    )

    # Apply decision thresholds
    df["decision"] = np.where(
        df["risk_score"] >= DECLINE_TH, "decline",
        np.where(df["risk_score"] >= REVIEW_TH, "review", "approve")
    )

    # Simple reason codes (rules) for explainability
    df["reason_code"] = "GENERAL_RISK"
    df.loc[df["velocity_1h"] >= 10, "reason_code"] = "HIGH_VELOCITY_1H"
    df.loc[df["ip_risk_score"] >= 90, "reason_code"] = "HIGH_IP_RISK"
    df.loc[df["account_age_days"] <= 7, "reason_code"] = "VERY_NEW_ACCOUNT"
    df.loc[df["amount_vs_avg_30d"] >= 4, "reason_code"] = "AMOUNT_SPIKE"
    df.loc[df["night_flag"] == 1, "reason_code"] = "NIGHT_TRANSACTION"

    # Output decisions file
    out_cols = [
        "transaction_id", "timestamp",
        "customer_id", "merchant_id", "merchant_category", "channel",
        "amount_gbp", "risk_score", "decision", "reason_code", "fraud_label"
    ]
    decisions = df[out_cols].copy()
    decisions.to_csv("outputs/decisions.csv", index=False)

    # Daily KPI table for monitoring
    decisions["date"] = decisions["timestamp"].dt.date

    daily = decisions.groupby("date").agg(
        transactions=("transaction_id", "count"),
        fraud_rate=("fraud_label", "mean"),
        approved=("decision", lambda x: (x == "approve").sum()),
        reviewed=("decision", lambda x: (x == "review").sum()),
        declined=("decision", lambda x: (x == "decline").sum()),
        caught_fraud=("fraud_label", lambda x: ((decisions.loc[x.index, "decision"] != "approve") &
                                               (decisions.loc[x.index, "fraud_label"] == 1)).sum()),
        missed_fraud=("fraud_label", lambda x: ((decisions.loc[x.index, "decision"] == "approve") &
                                               (decisions.loc[x.index, "fraud_label"] == 1)).sum()),
        false_declines=("fraud_label", lambda x: ((decisions.loc[x.index, "decision"] == "decline") &
                                                 (decisions.loc[x.index, "fraud_label"] == 0)).sum())
    ).reset_index()

    daily.to_csv("outputs/daily_kpis.csv", index=False)

    print("Created outputs/decisions.csv")
    print("Created outputs/daily_kpis.csv")
    print(f"Rows in decisions.csv: {len(decisions):,}")


if __name__ == "__main__":
    main()
