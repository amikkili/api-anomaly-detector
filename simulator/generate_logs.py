"""
MuleSoft API Log Simulator
===========================
Simulates realistic API logs as if coming from Anypoint Platform.
Generates normal traffic + injected anomalies (spikes, errors, floods).

Run: python simulator/generate_logs.py
"""

import csv
import random
import os
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — tweak these values to experiment
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_FILE   = "data/raw/api_logs.csv"
TOTAL_RECORDS = 5000       # 5000 rows = ~30 days of moderate API traffic
ANOMALY_RATE  = 0.05       # 5% anomalies — realistic for healthy APIs

# These are typical MuleSoft API endpoint names you'd see in Anypoint
API_ENDPOINTS = [
    "/api/v1/customers",
    "/api/v1/orders",
    "/api/v1/payments",
    "/api/v1/inventory",
    "/api/v2/products",
    "/api/v1/shipments",
    "/api/v1/auth/token",
    "/api/v2/invoices",
]

HTTP_METHODS = ["GET", "POST", "PUT", "DELETE"]


# ─────────────────────────────────────────────────────────────────────────────
# NORMAL RECORD
# random.gauss(180, 40) means: average 180ms, std deviation 40ms
# This creates a realistic bell curve of response times
# ─────────────────────────────────────────────────────────────────────────────
def normal_record(timestamp, endpoint, method):
    return {
        "timestamp"        : timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "endpoint"         : endpoint,
        "http_method"      : method,
        "status_code"      : random.choice([200, 200, 200, 201, 204]),  # weighted toward 200
        "response_time_ms" : round(random.gauss(180, 40), 2),
        "payload_size_kb"  : round(random.uniform(1, 50), 2),
        "client_ip"        : f"10.0.{random.randint(0,10)}.{random.randint(1,254)}",
        "is_anomaly"        : 0   # label for evaluation only — model won't see this
    }


# ─────────────────────────────────────────────────────────────────────────────
# ANOMALY RECORD
# Randomly picks 1 of 3 anomaly types and exaggerates the right fields
# ─────────────────────────────────────────────────────────────────────────────
def anomaly_record(timestamp, endpoint, method):
    anomaly_type = random.choice(["latency_spike", "error_burst", "traffic_flood"])

    base = {
        "timestamp"  : timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "endpoint"   : endpoint,
        "http_method": method,
        "client_ip"  : f"10.0.{random.randint(0,10)}.{random.randint(1,254)}",
        "is_anomaly"  : 1
    }

    if anomaly_type == "latency_spike":
        # Server is responding but extremely slowly
        base["status_code"]      = 200
        base["response_time_ms"] = round(random.uniform(2000, 8000), 2)
        base["payload_size_kb"]  = round(random.uniform(1, 50), 2)

    elif anomaly_type == "error_burst":
        # Server is throwing 5xx errors
        base["status_code"]      = random.choice([500, 503, 504, 429])
        base["response_time_ms"] = round(random.gauss(300, 60), 2)
        base["payload_size_kb"]  = round(random.uniform(0.1, 5), 2)

    elif anomaly_type == "traffic_flood":
        # Someone is sending massive payloads (DDoS or bad integration)
        base["status_code"]      = random.choice([200, 429])
        base["response_time_ms"] = round(random.gauss(400, 100), 2)
        base["payload_size_kb"]  = round(random.uniform(100, 500), 2)

    return base


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — wires everything together
# ─────────────────────────────────────────────────────────────────────────────
def generate_logs():
    os.makedirs("data/raw", exist_ok=True)

    # Start 30 days ago so we have historical data
    start_time = datetime.now() - timedelta(days=30)
    records    = []

    print(f"Generating {TOTAL_RECORDS} API log records...")

    for i in range(TOTAL_RECORDS):
        # Move timestamp forward slightly each iteration + add small random jitter
        # This simulates real traffic (not perfectly spaced)
        timestamp = start_time + timedelta(
            seconds = i * (30 * 24 * 3600 / TOTAL_RECORDS) + random.randint(-30, 30)
        )
        endpoint = random.choice(API_ENDPOINTS)
        method   = random.choice(HTTP_METHODS)

        if random.random() < ANOMALY_RATE:
            records.append(anomaly_record(timestamp, endpoint, method))
        else:
            records.append(normal_record(timestamp, endpoint, method))

    # Write all records to CSV
    fieldnames = [
        "timestamp", "endpoint", "http_method", "status_code",
        "response_time_ms", "payload_size_kb", "client_ip", "is_anomaly"
    ]
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    # Print summary
    anomaly_count = sum(1 for r in records if r["is_anomaly"] == 1)
    print(f"\nDone! File saved to: {OUTPUT_FILE}")
    print(f"   Total records  : {TOTAL_RECORDS}")
    print(f"   Normal records : {TOTAL_RECORDS - anomaly_count}")
    print(f"   Anomalies      : {anomaly_count} ({anomaly_count/TOTAL_RECORDS*100:.1f}%)")


if __name__ == "__main__":
    generate_logs()
