import pandas as pd
import numpy as np

PRED_PATH = "outputs/test_predictions.csv"

# Business costs
COST_FN = 500
COST_REVIEW = 3
COST_FALSE_DECLINE = 15


def calculate_cost(df, review_th, decline_th):

    decisions = np.where(
        df["risk_score"] >= decline_th, "decline",
        np.where(df["risk_score"] >= review_th, "review", "approve")
    )

    # False negatives (approved fraud)
    fn = ((decisions == "approve") & (df["fraud_label"] == 1)).sum()

    # Manual reviews
    reviews = (decisions == "review").sum()

    # False declines (good customers declined)
    false_declines = ((decisions == "decline") & (df["fraud_label"] == 0)).sum()

    total_cost = (
        fn * COST_FN +
        reviews * COST_REVIEW +
        false_declines * COST_FALSE_DECLINE
    )

    return total_cost


def main():

    print("Loading predictions...")
    df = pd.read_csv(PRED_PATH)

    best_cost = float("inf")
    best_review = None
    best_decline = None

    thresholds = np.arange(0.1, 0.95, 0.05)

    for review_th in thresholds:
        for decline_th in thresholds:

            if decline_th <= review_th:
                continue

            cost = calculate_cost(df, review_th, decline_th)

            if cost < best_cost:
                best_cost = cost
                best_review = review_th
                best_decline = decline_th

    print("Best thresholds found:")
    print(f"Review threshold: {best_review:.2f}")
    print(f"Decline threshold: {best_decline:.2f}")
    print(f"Total cost: £{best_cost:,.2f}")


if __name__ == "__main__":
    main()
