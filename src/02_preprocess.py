from __future__ import annotations

import os
import pandas as pd


RAW_PATH = "data/raw/uk_payments_synthetic_1_2m.csv"
OUT_DIR = "data/processed"

TRAIN_END_Q = 0.70
VALID_END_Q = 0.85


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Raw dataset not found at: {RAW_PATH}")

    print("Reading raw dataset...")
    df = pd.read_csv(RAW_PATH, parse_dates=["timestamp"])

    print("Sorting by timestamp...")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Basic validation
    required_cols = ["timestamp", "fraud_label"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    n = len(df)
    train_end = int(n * TRAIN_END_Q)
    valid_end = int(n * VALID_END_Q)

    train_df = df.iloc[:train_end].copy()
    valid_df = df.iloc[train_end:valid_end].copy()
    test_df  = df.iloc[valid_end:].copy()

    ensure_dir(OUT_DIR)

    train_path = os.path.join(OUT_DIR, "train.csv")
    valid_path = os.path.join(OUT_DIR, "valid.csv")
    test_path  = os.path.join(OUT_DIR, "test.csv")

    print("Writing splits...")
    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)
    test_df.to_csv(test_path, index=False)

    def rate(x: pd.DataFrame) -> float:
        return float(x["fraud_label"].mean())

    print("Done.")
    print(f"Train: {len(train_df):,} | fraud_rate={rate(train_df)*100:.2f}% | {train_df['timestamp'].min()} -> {train_df['timestamp'].max()}")
    print(f"Valid: {len(valid_df):,} | fraud_rate={rate(valid_df)*100:.2f}% | {valid_df['timestamp'].min()} -> {valid_df['timestamp'].max()}")
    print(f"Test : {len(test_df):,} | fraud_rate={rate(test_df)*100:.2f}% | {test_df['timestamp'].min()} -> {test_df['timestamp'].max()}")


if __name__ == "__main__":
    main()
