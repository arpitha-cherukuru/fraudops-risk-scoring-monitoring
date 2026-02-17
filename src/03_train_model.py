import os
import json
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler


TRAIN_PATH = "data/processed/train.csv"
VALID_PATH = "data/processed/valid.csv"
TEST_PATH  = "data/processed/test.csv"

OUT_DIR = "outputs"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_data(path):
    return pd.read_csv(path)


def main():

    print("Loading datasets...")
    train_df = load_data(TRAIN_PATH)
    valid_df = load_data(VALID_PATH)
    test_df  = load_data(TEST_PATH)

    # Select numeric features only (exclude IDs and timestamp)
    exclude_cols = [
        "transaction_id",
        "timestamp",
        "merchant_category",
        "channel",
        "fraud_label"
    ]

    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    X_train = train_df[feature_cols]
    y_train = train_df["fraud_label"]

    X_valid = valid_df[feature_cols]
    y_valid = valid_df["fraud_label"]

    X_test  = test_df[feature_cols]
    y_test  = test_df["fraud_label"]

    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled  = scaler.transform(X_test)

    print("Training Logistic Regression...")
    model = LogisticRegression(max_iter=200)
    model.fit(X_train_scaled, y_train)

    print("Evaluating model...")
    valid_probs = model.predict_proba(X_valid_scaled)[:, 1]
    test_probs  = model.predict_proba(X_test_scaled)[:, 1]

    roc_auc = roc_auc_score(y_test, test_probs)
    pr_auc  = average_precision_score(y_test, test_probs)

    print(f"Test ROC-AUC: {roc_auc:.4f}")
    print(f"Test PR-AUC: {pr_auc:.4f}")

    ensure_dir(OUT_DIR)

    # Save predictions
    output_df = test_df[["transaction_id", "timestamp"]].copy()
    output_df["fraud_label"] = y_test
    output_df["risk_score"] = test_probs

    output_df.to_csv("outputs/test_predictions.csv", index=False)

    # Save metrics
    metrics = {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc)
    }

    with open("outputs/model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Model training complete.")


if __name__ == "__main__":
    main()
