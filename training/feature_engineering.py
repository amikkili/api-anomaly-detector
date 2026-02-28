"""
Feature Engineering
====================
Transforms raw API logs into ML-ready features.

Adds rolling windows, error flags, time context, and deviation
signals — the meaningful patterns an ML model needs to detect
anomalies reliably.

Run: python training/feature_engineering.py
"""

import pandas as pd
import os

RAW_FILE       = "data/raw/api_logs.csv"
PROCESSED_FILE = "data/processed/features.csv"

# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features():

    # ── LOAD ──────────────────────────────────────────────────────────────────
    # parse_dates tells pandas to treat timestamp as a real datetime object
    # not just a plain string — this unlocks .dt.hour, .dt.dayofweek etc.
    print("Loading raw logs...")
    df = pd.read_csv(RAW_FILE, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  Loaded {len(df)} records across {df['endpoint'].nunique()} endpoints")

    # ── GROUP 1: ERROR SIGNALS ─────────────────────────────────────────────────
    # Simple binary flags — 1 or 0
    # These become especially powerful when fed into rolling averages below
    df["is_error"] = df["status_code"].apply(lambda x: 1 if x >= 500 else 0)
    df["is_4xx"]   = df["status_code"].apply(lambda x: 1 if 400 <= x < 500 else 0)

    # ── GROUP 2: ROLLING WINDOW STATS ─────────────────────────────────────────
    # We sort by endpoint + time so each endpoint's rolling window
    # is calculated independently — payments history doesn't pollute orders history
    df = df.sort_values(["endpoint", "timestamp"]).reset_index(drop=True)

    # window=10 means: look at the last 10 API calls for THIS endpoint
    # min_periods=1 handles the first few rows where we don't have 10 yet
    df["rolling_avg_response_ms"] = (
        df.groupby("endpoint")["response_time_ms"]
        .transform(lambda x: x.rolling(window=10, min_periods=1).mean())
    )

    df["rolling_error_rate"] = (
        df.groupby("endpoint")["is_error"]
        .transform(lambda x: x.rolling(window=10, min_periods=1).mean())
        # Result: 0.0 = no errors in last 10, 1.0 = all 10 were errors
    )

    df["rolling_avg_payload_kb"] = (
        df.groupby("endpoint")["payload_size_kb"]
        .transform(lambda x: x.rolling(window=10, min_periods=1).mean())
    )

    # ── GROUP 3: DEVIATION SIGNAL ──────────────────────────────────────────────
    # This is the single most powerful feature for latency spike detection
    # Example: rolling avg = 180ms, current call = 6000ms → deviation = 5820ms
    # The model learns: "large deviation = likely anomaly"
    df["response_time_deviation"] = (
        df["response_time_ms"] - df["rolling_avg_response_ms"]
    ).abs()

    # ── GROUP 4: TIME CONTEXT ──────────────────────────────────────────────────
    # Extract from the datetime object — only works because we used parse_dates above
    df["hour_of_day"] = df["timestamp"].dt.hour        # 0–23
    df["day_of_week"] = df["timestamp"].dt.dayofweek   # 0=Monday, 6=Sunday

    # ── GROUP 5: ENDPOINT IDENTITY ─────────────────────────────────────────────
    # ML models only understand numbers, not strings like "/api/v1/payments"
    # pd.factorize assigns a unique integer to each unique endpoint string
    # e.g. /api/v1/customers=0, /api/v1/orders=1, /api/v1/payments=2 etc.
    df["endpoint_id"] = pd.factorize(df["endpoint"])[0]

    # ── SELECT FINAL COLUMNS ───────────────────────────────────────────────────
    # These are the columns the model will train on
    # is_anomaly is kept ONLY for later evaluation — model won't use it in training
    feature_cols = [
        "endpoint_id",
        "response_time_ms",
        "payload_size_kb",
        "is_error",
        "is_4xx",
        "rolling_avg_response_ms",
        "rolling_error_rate",
        "rolling_avg_payload_kb",
        "response_time_deviation",
        "hour_of_day",
        "day_of_week",
        "is_anomaly",            # label — for evaluation only
    ]

    df_features = df[feature_cols].dropna()
    # dropna() removes any rows where rolling window couldn't compute
    # (very first rows of each endpoint group) — keeps data clean

    # ── SAVE ──────────────────────────────────────────────────────────────────
    os.makedirs("data/processed", exist_ok=True)
    df_features.to_csv(PROCESSED_FILE, index=False)

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    print(f"\nFeature engineering complete!")
    print(f"   Input rows   : {len(df)}")
    print(f"   Output rows  : {len(df_features)}  (difference = rows dropped by dropna)")
    print(f"   Features     : {len(feature_cols) - 1} ML features + 1 label column")
    print(f"   Saved to     : {PROCESSED_FILE}")

    print(f"\nFeature Stats:")
    print(df_features.describe().round(2).to_string())

    print(f"\nSample — Normal record:")
    print(df_features[df_features["is_anomaly"] == 0].iloc[0].to_string())

    print(f"\n Sample — Anomaly record:")
    print(df_features[df_features["is_anomaly"] == 1].iloc[0].to_string())


if __name__ == "__main__":
    engineer_features()