from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


@dataclass
class Config:
    # Scale
    n_transactions: int = 1_200_000

    # Timeline
    start_ts: str = "2025-01-01 00:00:00"   # change if you want
    freq_minutes: int = 1                  # 1 transaction per minute baseline, with bursts simulated

    # Population sizes (controls uniqueness + realism)
    n_customers: int = 120_000
    n_merchants: int = 12_000
    n_devices: int = 160_000

    # Fraud controls
    base_fraud_rate: float = 0.012         # ~1.2% base fraud
    drift_day: int = 120                   # behaviour shifts after 120 days

    # Output
    out_path: str = "data/raw/uk_payments_synthetic_1_2m.csv"


def ensure_dirs(path: str) -> None:
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def main() -> None:
    cfg = Config()
    rng = np.random.default_rng(42)

    # -----------------------------
    # 1) Generate timestamps (vectorised)
    # -----------------------------
    start = pd.Timestamp(cfg.start_ts)
    # Create time index with minute resolution, then add random seconds jitter
    # This gives realistic intra-minute variation.
    base_minutes = pd.date_range(start=start, periods=cfg.n_transactions, freq=f"{cfg.freq_minutes}min")
    seconds_jitter = rng.integers(0, 60, size=cfg.n_transactions)
    timestamps = (base_minutes + pd.to_timedelta(seconds_jitter, unit="s")).astype("datetime64[ns]")

    hour = pd.DatetimeIndex(timestamps).hour.values
    day_of_week = pd.DatetimeIndex(timestamps).dayofweek.values  # 0=Mon ... 6=Sun
    weekend_flag = (day_of_week >= 5).astype(np.int8)
    night_flag = ((hour <= 5) | (hour >= 23)).astype(np.int8)

    # Day index for drift simulation
    day_index = ((timestamps - timestamps[0]) / np.timedelta64(1, "D")).astype(int)
    drift_flag = (day_index >= cfg.drift_day).astype(np.int8)

    # -----------------------------
    # 2) Entities: customers, merchants, devices
    # -----------------------------
    customer_id = rng.integers(1, cfg.n_customers + 1, size=cfg.n_transactions)
    merchant_id = rng.integers(1, cfg.n_merchants + 1, size=cfg.n_transactions)
    device_id = rng.integers(1, cfg.n_devices + 1, size=cfg.n_transactions)

    # -----------------------------
    # 3) Merchant categories and channels (UK-feel)
    # -----------------------------
    merchant_categories = np.array(
        ["grocery", "fuel", "fashion", "electronics", "travel", "restaurants", "utilities", "gaming", "crypto", "pharmacy"],
        dtype=object,
    )
    # Category distribution: more grocery/fuel/utilities, fewer crypto
    cat_probs = np.array([0.20, 0.12, 0.12, 0.10, 0.08, 0.13, 0.12, 0.08, 0.03, 0.02])
    merchant_category = rng.choice(merchant_categories, size=cfg.n_transactions, p=cat_probs)

    channels = np.array(["card_present", "ecommerce", "transfer"], dtype=object)
    # UK payments: card_present still big, ecommerce big, transfers smaller
    channel_probs = np.array([0.45, 0.45, 0.10])
    channel = rng.choice(channels, size=cfg.n_transactions, p=channel_probs)

    # -----------------------------
    # 4) Amount generation (category + channel effects)
    # -----------------------------
    # Base amounts from lognormal/exponential mix for realistic long tail.
    base_amt = rng.lognormal(mean=3.0, sigma=0.8, size=cfg.n_transactions)  # around ~20-30 typical, tail higher
    base_amt = np.clip(base_amt, 0.5, None)

    # Category multipliers (travel/electronics higher)
    cat_mult = np.ones(cfg.n_transactions, dtype=np.float32)
    cat_mult[merchant_category == "travel"] = 2.3
    cat_mult[merchant_category == "electronics"] = 1.9
    cat_mult[merchant_category == "crypto"] = 2.8
    cat_mult[merchant_category == "utilities"] = 1.2
    cat_mult[merchant_category == "fuel"] = 1.1
    cat_mult[merchant_category == "grocery"] = 1.0

    # Channel multipliers (ecommerce slightly higher variance)
    ch_mult = np.ones(cfg.n_transactions, dtype=np.float32)
    ch_mult[channel == "ecommerce"] = 1.15
    ch_mult[channel == "transfer"] = 1.35

    amount = base_amt * cat_mult * ch_mult
    amount = np.round(amount, 2)

    # -----------------------------
    # 5) Risk features: ip risk, distance, account age, chargeback history
    # -----------------------------
    # IP risk: 0-100, ecommerce tends higher
    ip_risk_score = rng.integers(0, 101, size=cfg.n_transactions)
    ip_risk_score = ip_risk_score + (channel == "ecommerce") * rng.integers(0, 15, size=cfg.n_transactions)
    ip_risk_score = np.clip(ip_risk_score, 0, 100).astype(np.int16)

    # Distance from home: card_present lower, ecommerce higher
    distance_from_home_km = rng.gamma(shape=2.0, scale=5.0, size=cfg.n_transactions)  # mostly small
    distance_from_home_km = distance_from_home_km + (channel == "ecommerce") * rng.gamma(2.0, 15.0, size=cfg.n_transactions)
    distance_from_home_km = np.round(distance_from_home_km, 2)

    # Account age in days: skewed towards older accounts but with many new accounts
    # Drift: after drift_day, fraudsters use more new accounts (shift distribution)
    base_account_age = rng.integers(1, 2000, size=cfg.n_transactions)  # up to ~5.5 years
    new_account_boost = (drift_flag == 1) * rng.integers(0, 200, size=cfg.n_transactions)
    account_age_days = np.clip(base_account_age - new_account_boost, 1, None).astype(np.int16)

    # Chargeback history count: small integers, correlated with higher risk
    chargeback_history_count = rng.poisson(lam=0.15, size=cfg.n_transactions).astype(np.int16)
    chargeback_history_count += (ip_risk_score > 80).astype(np.int16) * rng.poisson(0.3, size=cfg.n_transactions).astype(np.int16)
    chargeback_history_count = np.clip(chargeback_history_count, 0, 10).astype(np.int16)

    # -----------------------------
    # 6) Velocity features (approximation, realistic enough + fast)
    # -----------------------------
    # We approximate burstiness using a mixture distribution:
    # - Most transactions have low velocity
    # - Some have medium bursts
    # - A small fraction have high bursts (fraud-y)
    mix = rng.random(cfg.n_transactions)
    velocity_1h = np.where(
        mix < 0.90, rng.poisson(1.2, size=cfg.n_transactions),
        np.where(mix < 0.98, rng.poisson(4.0, size=cfg.n_transactions),
                 rng.poisson(10.0, size=cfg.n_transactions))
    ).astype(np.int16)

    velocity_24h = (velocity_1h + rng.poisson(3.0, size=cfg.n_transactions)).astype(np.int16)
    velocity_24h = np.clip(velocity_24h, 0, 60).astype(np.int16)

    # -----------------------------
    # 7) Customer baseline behaviour (avg_30d + amount_vs_avg)
    # -----------------------------
    # Create a customer "typical spend" baseline, then measure deviation.
    customer_base_spend = rng.lognormal(mean=3.0, sigma=0.6, size=cfg.n_customers + 1)  # index by customer_id
    avg_amount_30d = customer_base_spend[customer_id] * rng.uniform(0.7, 1.3, size=cfg.n_transactions)
    avg_amount_30d = np.round(avg_amount_30d, 2)

    amount_vs_avg_30d = amount / np.clip(avg_amount_30d, 0.01, None)
    amount_vs_avg_30d = np.round(amount_vs_avg_30d, 2)

    # -----------------------------
    # 8) Fraud label generation (realistic + drift)
    # -----------------------------
    # Risk score components (log-odds style).
    # Drift behaviour: after drift_day, fraud shifts towards:
    # - Lower amount spikes but higher velocity + ecommerce + new accounts + high ip risk
    amount_component = np.where(amount > 250, 0.7, 0.0) + np.where(amount > 800, 0.8, 0.0)
    amount_component = amount_component * (1 - drift_flag) + (amount_component * 0.65) * drift_flag  # reduce amount signal after drift

    velocity_component = np.where(velocity_1h >= 6, 0.9, 0.0) + np.where(velocity_1h >= 10, 0.7, 0.0)
    velocity_component = velocity_component + 0.25 * drift_flag  # stronger after drift

    ip_component = np.where(ip_risk_score >= 75, 1.0, 0.0) + np.where(ip_risk_score >= 90, 0.7, 0.0)
    ip_component = ip_component + 0.15 * (channel == "ecommerce").astype(np.float32)

    new_account_component = np.where(account_age_days <= 30, 0.9, 0.0) + np.where(account_age_days <= 7, 0.7, 0.0)
    new_account_component = new_account_component + 0.2 * drift_flag  # more new-account abuse after drift

    distance_component = np.where(distance_from_home_km >= 40, 0.45, 0.0) + np.where(distance_from_home_km >= 120, 0.55, 0.0)

    time_component = 0.25 * night_flag + 0.12 * weekend_flag

    category_component = np.zeros(cfg.n_transactions, dtype=np.float32)
    category_component[merchant_category == "electronics"] = 0.25
    category_component[merchant_category == "travel"] = 0.30
    category_component[merchant_category == "gaming"] = 0.25
    category_component[merchant_category == "crypto"] = 0.60

    channel_component = np.zeros(cfg.n_transactions, dtype=np.float32)
    channel_component[channel == "ecommerce"] = 0.35
    channel_component[channel == "transfer"] = 0.25

    chargeback_component = 0.18 * np.clip(chargeback_history_count, 0, 6)

    # Combine to log-odds
    log_odds = (
        -4.4  # base intercept -> controls overall fraud rate
        + amount_component
        + velocity_component
        + ip_component
        + new_account_component
        + distance_component
        + time_component
        + category_component
        + channel_component
        + chargeback_component
    )

    # Adjust to target base fraud rate roughly
    # (This is a light correction; exact rate will vary slightly.)
    log_odds = log_odds + np.log(cfg.base_fraud_rate / (1 - cfg.base_fraud_rate)) - (-4.4)

    prob_fraud = sigmoid(log_odds)
    fraud_label = (rng.random(cfg.n_transactions) < prob_fraud).astype(np.int8)

    # -----------------------------
    # 9) Assemble DataFrame
    # -----------------------------
    df = pd.DataFrame(
        {
            "transaction_id": np.arange(1, cfg.n_transactions + 1, dtype=np.int64),
            "timestamp": pd.to_datetime(timestamps),
            "customer_id": customer_id.astype(np.int32),
            "merchant_id": merchant_id.astype(np.int32),
            "merchant_category": merchant_category,
            "channel": channel,
            "amount_gbp": amount.astype(np.float32),
            "ip_risk_score": ip_risk_score,
            "distance_from_home_km": distance_from_home_km.astype(np.float32),
            "account_age_days": account_age_days,
            "chargeback_history_count": chargeback_history_count,
            "velocity_1h": velocity_1h,
            "velocity_24h": velocity_24h,
            "avg_amount_30d": avg_amount_30d.astype(np.float32),
            "amount_vs_avg_30d": amount_vs_avg_30d.astype(np.float32),
            "night_flag": night_flag,
            "weekend_flag": weekend_flag,
            "drift_flag": drift_flag,
            "fraud_label": fraud_label,
        }
    )

    # -----------------------------
    # 10) Save
    # -----------------------------
    ensure_dirs(cfg.out_path)
    df.to_csv(cfg.out_path, index=False)

    # Quick console summary
    fraud_rate = df["fraud_label"].mean()
    print(f"Saved: {cfg.out_path}")
    print(f"Rows: {len(df):,}")
    print(f"Fraud rate: {fraud_rate:.4f} ({fraud_rate*100:.2f}%)")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")


if __name__ == "__main__":
    main()
