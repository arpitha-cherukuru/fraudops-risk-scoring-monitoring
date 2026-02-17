import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

TRAIN_PATH = "data/processed/train.csv"
OUT_PATH = "outputs/model_feature_importance.csv"


def main():
    print("Loading training data...")
    df = pd.read_csv(TRAIN_PATH)

    exclude_cols = [
        "transaction_id",
        "timestamp",
        "merchant_category",
        "channel",
        "fraud_label"
    ]

    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols]
    y = df["fraud_label"]

    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Training logistic regression (for explainability)...")
    model = LogisticRegression(max_iter=200)
    model.fit(X_scaled, y)

    coefs = model.coef_[0]

    importance = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": coefs,
        "abs_importance": np.abs(coefs)
    }).sort_values("abs_importance", ascending=False)

    importance.to_csv(OUT_PATH, index=False)

    print("Created outputs/model_feature_importance.csv")
    print("Top 10 features:")
    print(importance.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
