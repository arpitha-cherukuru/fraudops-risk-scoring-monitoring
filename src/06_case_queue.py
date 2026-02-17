import pandas as pd
import numpy as np

DECISIONS_PATH = "outputs/decisions.csv"
OUT_PATH = "outputs/case_queue.csv"

DAILY_REVIEW_CAPACITY = 40
SLA_DAYS = 1


def main():
    df = pd.read_csv(DECISIONS_PATH, parse_dates=["timestamp"])
    df["date"] = df["timestamp"].dt.date

    # Only review cases go into queue
    queue = df[df["decision"] == "review"].copy()

    # Keep only what we need + stable identifier
    queue = queue[[
        "transaction_id", "timestamp", "date",
        "risk_score", "fraud_label",
        "customer_id", "merchant_id", "merchant_category", "channel", "amount_gbp", "reason_code"
    ]].copy()

    # Sort by arrival date then highest risk first
    queue = queue.sort_values(["date", "risk_score"], ascending=[True, False]).reset_index(drop=True)

    # Tracking columns
    queue["reviewed"] = 0
    queue["review_date"] = pd.NaT
    queue["backlog_days"] = 0
    queue["sla_breach"] = 0

    # We will simulate day-by-day
    all_dates = sorted(queue["date"].unique())

    # Backlog stores transaction_ids waiting for review
    backlog_ids = []

    # For fast lookup during updates
    id_to_row = {tid: i for i, tid in enumerate(queue["transaction_id"].values)}

    for current_date in all_dates:
        # Add today’s incoming review cases to backlog
        today_ids = queue.loc[queue["date"] == current_date, "transaction_id"].tolist()
        backlog_ids.extend(today_ids)

        if len(backlog_ids) == 0:
            continue

        # Prioritise by risk_score (highest first)
        backlog_slice = queue.loc[queue["transaction_id"].isin(backlog_ids), ["transaction_id", "risk_score", "date"]]
        backlog_slice = backlog_slice.sort_values("risk_score", ascending=False)

        # Select up to capacity for today
        to_review = backlog_slice.head(DAILY_REVIEW_CAPACITY)["transaction_id"].tolist()

        # Mark reviewed
        review_dt = pd.to_datetime(current_date)
        for tid in to_review:
            row_idx = id_to_row[tid]
            queue.at[row_idx, "reviewed"] = 1
            queue.at[row_idx, "review_date"] = review_dt

        # Remove reviewed from backlog
        to_review_set = set(to_review)
        backlog_ids = [tid for tid in backlog_ids if tid not in to_review_set]

        # Update backlog days and SLA breaches for remaining backlog
        if len(backlog_ids) > 0:
            remaining = queue.loc[queue["transaction_id"].isin(backlog_ids), ["transaction_id", "date"]].copy()
            remaining["backlog_days"] = (review_dt - pd.to_datetime(remaining["date"])).dt.days

            # Update those values back into queue
            for tid, days in zip(remaining["transaction_id"].values, remaining["backlog_days"].values):
                row_idx = id_to_row[tid]
                queue.at[row_idx, "backlog_days"] = int(days)
                queue.at[row_idx, "sla_breach"] = 1 if days > SLA_DAYS else 0

    queue.to_csv(OUT_PATH, index=False)

    print("Created outputs/case_queue.csv")
    print(f"Total review cases: {len(queue):,}")
    print(f"Reviewed (within simulation): {(queue['reviewed'] == 1).sum():,}")
    print(f"SLA breaches: {(queue['sla_breach'] == 1).sum():,}")


if __name__ == "__main__":
    main()
