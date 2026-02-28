"""
FastAPI Model Server
=====================
Exposes the trained Isolation Forest model as a production REST API.
Any system — MuleSoft, Postman, curl, another Python script — can call this.

Endpoints:
  GET  /health         -> Check if model is loaded
  POST /predict        -> Score a single API call
  POST /predict/batch  -> Score multiple API calls at once
  GET  /docs           -> Auto-generated Swagger UI (FastAPI magic!)

Run locally:
  uvicorn serving.app:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import pickle
import numpy as np
import os
from datetime import datetime

# -----------------------------------------------------------------------------
# APP SETUP
# FastAPI creates the app + auto-generates /docs Swagger UI for free
# -----------------------------------------------------------------------------
app = FastAPI(
    title       = "API Anomaly Detector",
    description = "ML-powered anomaly detection for MuleSoft API monitoring",
    version     = "1.0.0"
)

# -----------------------------------------------------------------------------
# MODEL LOADING
# Both model AND scaler must be loaded together
# The scaler MUST transform input before model sees it — same as during training
# If scaler is missing, model gets unscaled data and gives wrong predictions
# -----------------------------------------------------------------------------
MODEL_DIR   = "serving/model"
model       = None
scaler      = None

@app.on_event("startup")
def load_model():
    global model, scaler
    model_path  = os.path.join(MODEL_DIR, "isolation_forest.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

    try:
        with open(model_path,  "rb") as f: model  = pickle.load(f)
        with open(scaler_path, "rb") as f: scaler = pickle.load(f)
        print("[READY] Model and scaler loaded successfully")
    except FileNotFoundError as e:
        print(f"[ERROR] Model file not found: {e}")
        print("[ERROR] Run training/train_model.py first!")


# -----------------------------------------------------------------------------
# REQUEST SCHEMA — Pydantic model
# This is like a DataWeave schema definition — it validates every field
# automatically before your code even runs. Wrong type? FastAPI rejects it.
# Field(...) means required. Field(0.0, ge=0) means optional with a minimum.
# -----------------------------------------------------------------------------
class APIMetrics(BaseModel):
    endpoint_id              : int   = Field(..., ge=0, le=7,    description="0-7 mapped to your 8 endpoints")
    response_time_ms         : float = Field(..., ge=0,          description="Response time in milliseconds")
    payload_size_kb          : float = Field(..., ge=0,          description="Payload size in kilobytes")
    is_error                 : int   = Field(..., ge=0, le=1,    description="1 if HTTP status >= 500")
    is_4xx                   : int   = Field(..., ge=0, le=1,    description="1 if HTTP status 400-499")
    rolling_avg_response_ms  : float = Field(..., ge=0,          description="Rolling avg response time last 10 calls")
    rolling_error_rate       : float = Field(..., ge=0.0, le=1.0,description="Error rate last 10 calls (0.0 to 1.0)")
    rolling_avg_payload_kb   : float = Field(..., ge=0,          description="Rolling avg payload last 10 calls")
    response_time_deviation  : float = Field(..., ge=0,          description="Abs deviation from rolling avg")
    hour_of_day              : int   = Field(..., ge=0, le=23,   description="Hour of day 0-23")
    day_of_week              : int   = Field(..., ge=0, le=6,    description="Day of week 0=Monday 6=Sunday")

    # Example shown in Swagger UI — makes testing much easier
    class Config:
        json_schema_extra = {
            "example": {
                "endpoint_id"             : 2,
                "response_time_ms"        : 185.3,
                "payload_size_kb"         : 22.4,
                "is_error"                : 0,
                "is_4xx"                  : 0,
                "rolling_avg_response_ms" : 178.5,
                "rolling_error_rate"      : 0.0,
                "rolling_avg_payload_kb"  : 20.1,
                "response_time_deviation" : 6.8,
                "hour_of_day"             : 14,
                "day_of_week"             : 1
            }
        }


# -----------------------------------------------------------------------------
# RESPONSE SCHEMA
# Consistent, documented response structure — like an API contract
# -----------------------------------------------------------------------------
class PredictionResponse(BaseModel):
    is_anomaly    : int
    anomaly_score : float
    risk_level    : str
    message       : str
    timestamp     : str


class BatchPredictionResponse(BaseModel):
    total_records    : int
    anomaly_count    : int
    anomaly_rate_pct : float
    results          : List[PredictionResponse]


# -----------------------------------------------------------------------------
# HELPER — Converts raw model output to a human-readable risk level
# Isolation Forest returns negative scores for anomalies
# We flip and normalize so: higher score = more anomalous
# -----------------------------------------------------------------------------
def compute_risk(raw_score: float, is_anomaly: int) -> tuple:
    # decision_function gives negative = anomaly, positive = normal
    # We normalize to 0-1 range where 1 = most anomalous
    normalized = float(np.clip((raw_score * -1 + 0.5), 0, 1))

    if not is_anomaly:
        return normalized, "LOW", "Normal API behavior detected"
    elif normalized < 0.6:
        return normalized, "MEDIUM", "Slight anomaly detected - monitor closely"
    elif normalized < 0.8:
        return normalized, "HIGH", "Anomaly detected - investigation recommended"
    else:
        return normalized, "CRITICAL", "Severe anomaly detected - immediate attention required"


# -----------------------------------------------------------------------------
# ENDPOINT 1 — Health Check
# Always build this first — tells you if the server + model are ready
# MuleSoft can poll this before sending real traffic
# -----------------------------------------------------------------------------
@app.get("/health")
def health_check():
    model_loaded = model is not None and scaler is not None
    return {
        "status"       : "healthy" if model_loaded else "unhealthy",
        "model_loaded" : model_loaded,
        "timestamp"    : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version"      : "1.0.0"
    }


# -----------------------------------------------------------------------------
# ENDPOINT 2 — Single Prediction
# The main endpoint — scores one API call's metrics
# Returns anomaly score, risk level, and human-readable message
# -----------------------------------------------------------------------------
@app.post("/predict", response_model=PredictionResponse)
def predict(metrics: APIMetrics):
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run training/train_model.py first."
        )

    # Build feature array in EXACT same order as training
    # Order matters — wrong order = wrong predictions silently
    feature_order = [
        metrics.endpoint_id,
        metrics.response_time_ms,
        metrics.payload_size_kb,
        metrics.is_error,
        metrics.is_4xx,
        metrics.rolling_avg_response_ms,
        metrics.rolling_error_rate,
        metrics.rolling_avg_payload_kb,
        metrics.response_time_deviation,
        metrics.hour_of_day,
        metrics.day_of_week,
    ]

    X = np.array(feature_order).reshape(1, -1)     # shape: (1, 11)
    X_scaled = scaler.transform(X)                  # MUST scale before predicting

    raw_pred  = model.predict(X_scaled)[0]          # -1 or +1
    raw_score = model.decision_function(X_scaled)[0]# continuous score

    is_anomaly              = int(raw_pred == -1)
    anomaly_score, risk, msg = compute_risk(raw_score, is_anomaly)

    return PredictionResponse(
        is_anomaly    = is_anomaly,
        anomaly_score = round(anomaly_score, 4),
        risk_level    = risk,
        message       = msg,
        timestamp     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )


# -----------------------------------------------------------------------------
# ENDPOINT 3 — Batch Prediction
# Send multiple records at once — efficient for bulk monitoring
# MuleSoft could batch the last 100 API calls and send them all at once
# -----------------------------------------------------------------------------
@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(records: List[APIMetrics]):
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run training/train_model.py first."
        )

    if len(records) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Batch size too large. Maximum 1000 records per request."
        )

    results = []
    for metrics in records:
        feature_order = [
            metrics.endpoint_id,
            metrics.response_time_ms,
            metrics.payload_size_kb,
            metrics.is_error,
            metrics.is_4xx,
            metrics.rolling_avg_response_ms,
            metrics.rolling_error_rate,
            metrics.rolling_avg_payload_kb,
            metrics.response_time_deviation,
            metrics.hour_of_day,
            metrics.day_of_week,
        ]

        X        = np.array(feature_order).reshape(1, -1)
        X_scaled = scaler.transform(X)
        raw_pred  = model.predict(X_scaled)[0]
        raw_score = model.decision_function(X_scaled)[0]

        is_anomaly               = int(raw_pred == -1)
        anomaly_score, risk, msg = compute_risk(raw_score, is_anomaly)

        results.append(PredictionResponse(
            is_anomaly    = is_anomaly,
            anomaly_score = round(anomaly_score, 4),
            risk_level    = risk,
            message       = msg,
            timestamp     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))

    anomaly_count = sum(r.is_anomaly for r in results)

    return BatchPredictionResponse(
        total_records    = len(results),
        anomaly_count    = anomaly_count,
        anomaly_rate_pct = round(anomaly_count / len(results) * 100, 2),
        results          = results
    )