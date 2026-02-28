"""
Model Training — Isolation Forest + MLflow
==========================================
Trains an unsupervised anomaly detection model on engineered API features.

Key concepts:
  - Isolation Forest: no labels needed, learns "normal" patterns
  - StandardScaler: normalizes features so no single feature dominates
  - MLflow: tracks every experiment run locally for comparison
  - contamination: the only hyperparameter you need to tune (expected anomaly %)

Run: python training/train_model.py
"""

import pandas as pd
import numpy as np
import pickle
import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────
PROCESSED_FILE = "data/processed/features.csv"
MODEL_DIR      = "serving/model"
MLFLOW_URI     = "mlruns"   # local folder — no server needed

# These are the exact columns feature_engineering.py produced
# We explicitly list them so the order is always guaranteed
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

# ─────────────────────────────────────────────────────────────────────────────
# WHY SCALE FEATURES?
# response_time_ms can be 0–8000
# rolling_error_rate is always 0.0–1.0
# Without scaling, response_time_ms dominates — the model ignores small features
# StandardScaler brings EVERY feature to mean=0, std=1 — equal playing field
# ─────────────────────────────────────────────────────────────────────────────
def scale_features(X_train):
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    return scaler, X_scaled


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION HELPER
# Isolation Forest outputs: -1 (anomaly) or +1 (normal)
# We convert to: 1 (anomaly) or 0 (normal) — matches our is_anomaly label
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(y_true, raw_predictions):
    y_pred = np.where(raw_predictions == -1, 1, 0)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    accuracy  = np.mean(y_true == y_pred)
    cm        = confusion_matrix(y_true, y_pred)

    print(f"\n Model Evaluation Results:")
    print(f"   Accuracy  : {accuracy:.3f}")
    print(f"   Precision : {precision:.3f}")
    print(f"   Recall    : {recall:.3f}")
    print(f"   F1 Score  : {f1:.3f}")
    print(f"\n   Confusion Matrix:")
    print(f"                  Predicted")
    print(f"               Normal   Anomaly")
    print(f"   Actual Normal  {cm[0][0]:4d}      {cm[0][1]:4d}   <- TN / FP")
    print(f"   Actual Anomaly {cm[1][0]:4d}      {cm[1][1]:4d}   <- FN / TP")
    print(f"\n   TN (correct normal)  : {cm[0][0]}  — normal calls correctly ignored")
    print(f"   TP (caught anomalies): {cm[1][1]}  — real anomalies we caught")
    print(f"   FP (false alarms)    : {cm[0][1]}  — normal calls wrongly flagged")
    print(f"   FN (missed anomalies): {cm[1][0]}  — real anomalies we missed")

    return accuracy, precision, recall, f1, y_pred


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def train():

    # ── Setup MLflow ──────────────────────────────────────────────────────────
    # set_tracking_uri tells MLflow to save everything in a local "mlruns" folder
    # set_experiment groups related runs — like branches in Git
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("api-anomaly-detection")

    # ── Load features ─────────────────────────────────────────────────────────
    print("Loading engineered features...")
    df     = pd.read_csv(PROCESSED_FILE)
    X      = df[FEATURE_COLS].values   # feature matrix  — shape: (5000, 11)
    y_true = df["is_anomaly"].values   # labels for eval  — shape: (5000,)

    print(f"  Dataset shape  : {X.shape}")
    print(f"  Normal records : {(y_true == 0).sum()}")
    print(f"  Anomaly records: {(y_true == 1).sum()} ({y_true.mean()*100:.1f}%)")

    # ── Scale features ────────────────────────────────────────────────────────
    print("\n Scaling features with StandardScaler...")
    scaler, X_scaled = scale_features(X)
    print("  Done all features now have mean~0, std~1")

    # ── Hyperparameters ────────────────────────────────────────────────────────
    # contamination = what % of your data you expect to be anomalies
    # This should match your real-world anomaly rate
    # TIP: Try 0.03, 0.05, 0.08 in separate runs and compare in MLflow UI
    contamination = 0.05
    n_estimators  = 100    # number of trees — more = more stable, but slower
    max_features  = 1.0    # fraction of features each tree sees

    # ── Start MLflow run ──────────────────────────────────────────────────────
    # Everything inside this block gets tracked automatically
    with mlflow.start_run(run_name="isolation_forest_v1"):

        # Log hyperparameters — these appear in MLflow UI for comparison
        mlflow.log_param("contamination",  contamination)
        mlflow.log_param("n_estimators",   n_estimators)
        mlflow.log_param("max_features",   max_features)
        mlflow.log_param("dataset_size",   len(df))
        mlflow.log_param("feature_count",  len(FEATURE_COLS))
        mlflow.log_param("scaler",         "StandardScaler")

        # ── Train ──────────────────────────────────────────────────────────────
        print(f"\nTraining Isolation Forest...")
        print(f"  contamination = {contamination}  (expected anomaly %)")
        print(f"  n_estimators  = {n_estimators}  (number of trees)")

        model = IsolationForest(
            contamination = contamination,
            n_estimators  = n_estimators,
            max_features  = max_features,
            random_state  = 42,     # makes results reproducible across runs
            n_jobs        = -1      # use all CPU cores — faster training
        )
        model.fit(X_scaled)
        print("  Training complete")

        # ── Predict & Evaluate ─────────────────────────────────────────────────
        raw_predictions = model.predict(X_scaled)
        accuracy, precision, recall, f1, y_pred = evaluate(y_true, raw_predictions)

        # Log metrics to MLflow — these appear as columns in the UI
        mlflow.log_metric("accuracy",  round(accuracy,  4))
        mlflow.log_metric("precision", round(precision, 4))
        mlflow.log_metric("recall",    round(recall,    4))
        mlflow.log_metric("f1_score",  round(f1,        4))

        # ── Save model + scaler to disk ────────────────────────────────────────
        # Both must be saved together — scaler must transform input
        # BEFORE the model sees it, every single time
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path  = os.path.join(MODEL_DIR, "isolation_forest.pkl")
        scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

        with open(model_path,  "wb") as f: pickle.dump(model,  f)
        with open(scaler_path, "wb") as f: pickle.dump(scaler, f)

        # Log model artifact to MLflow — stores a copy inside mlruns/
        mlflow.sklearn.log_model(model, artifact_path="isolation_forest")
        mlflow.log_artifact(scaler_path)

        run_id = mlflow.active_run().info.run_id

        print(f"\n[DONE] Model saved  -> {model_path}")
        print(f"[DONE] Scaler saved -> {scaler_path}")
        print(f"[DONE] MLflow Run ID: {run_id}")

    # ── Feature importance (anomaly score contribution) ────────────────────────
    # Isolation Forest doesn't give traditional feature importance
    # But we can approximate it by checking score variance per feature
    print(f"\nApproximate Feature Influence (score variance):")
    importances = []
    for i, col in enumerate(FEATURE_COLS):
        X_single        = np.zeros_like(X_scaled)
        X_single[:, i]  = X_scaled[:, i]
        scores          = model.decision_function(X_single)
        importances.append((col, round(float(np.std(scores)), 4)))

    importances.sort(key=lambda x: x[1], reverse=True)
    for rank, (col, score) in enumerate(importances, 1):
        bar = "|" * int(score * 40)
        print(f"  {rank:2d}. {col:30s} {bar} {score}")

    print(f"\nTip: Run 'mlflow ui' in terminal to compare experiment runs visually")
    print(f"        Then open: http://localhost:5000")


if __name__ == "__main__":
    train()