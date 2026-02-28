import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

PROCESSED_FILE = "data/processed/features.csv"
MODEL_DIR      = "serving/model"

FEATURE_COLS = [
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
]

def scale_features(X):
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return scaler, X_scaled

def evaluate(y_true, raw_predictions):
    y_pred    = np.where(raw_predictions == -1, 1, 0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    accuracy  = np.mean(y_true == y_pred)
    cm        = confusion_matrix(y_true, y_pred)

    print(f"\nModel Evaluation Results:")
    print(f"  Accuracy  : {accuracy:.3f}")
    print(f"  Precision : {precision:.3f}")
    print(f"  Recall    : {recall:.3f}")
    print(f"  F1 Score  : {f1:.3f}")
    print(f"\n  Confusion Matrix:")
    print(f"                 Predicted")
    print(f"              Normal   Anomaly")
    print(f"  Actual Normal  {cm[0][0]:4d}     {cm[0][1]:4d}  <- TN / FP")
    print(f"  Actual Anomaly {cm[1][0]:4d}     {cm[1][1]:4d}  <- FN / TP")

    return accuracy, precision, recall, f1, y_pred

def train():
    print("Loading engineered features...")
    df     = pd.read_csv(PROCESSED_FILE)
    X      = df[FEATURE_COLS].values
    y_true = df["is_anomaly"].values

    print(f"  Dataset shape  : {X.shape}")
    print(f"  Normal records : {(y_true == 0).sum()}")
    print(f"  Anomaly records: {(y_true == 1).sum()} ({y_true.mean()*100:.1f}%)")

    print("\nScaling features with StandardScaler...")
    scaler, X_scaled = scale_features(X)
    print("  Done - all features now have mean~0, std~1")

    contamination = 0.05
    n_estimators  = 100
    max_features  = 1.0

    print(f"\nTraining Isolation Forest...")
    print(f"  contamination = {contamination}")
    print(f"  n_estimators  = {n_estimators}")

    model = IsolationForest(
        contamination = contamination,
        n_estimators  = n_estimators,
        max_features  = max_features,
        random_state  = 42,
        n_jobs        = -1
    )
    model.fit(X_scaled)
    print("  Training complete")

    raw_predictions = model.predict(X_scaled)
    accuracy, precision, recall, f1, y_pred = evaluate(y_true, raw_predictions)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path  = os.path.join(MODEL_DIR, "isolation_forest.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

    with open(model_path,  "wb") as f: pickle.dump(model,  f)
    with open(scaler_path, "wb") as f: pickle.dump(scaler, f)

    print(f"\n[DONE] Model saved  -> {model_path}")
    print(f"[DONE] Scaler saved -> {scaler_path}")
    print(f"[DONE] Accuracy: {accuracy:.3f}  Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")


if __name__ == "__main__":
    train()