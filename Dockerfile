FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all needed folders
COPY serving/ ./serving/
COPY training/ ./training/
COPY simulator/ ./simulator/

# Create data directories
RUN mkdir -p data/raw data/processed

# Step 1 — Generate fake logs
# Step 2 — Engineer features  
# Step 3 — Train model
RUN python simulator/generate_logs.py && \
    python training/feature_engineering.py && \
    python training/train_model.py

ENV PYTHONUNBUFFERED=1
ENV PYTHONUTF8=1

CMD ["sh", "-c", "uvicorn serving.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
```

---

## Also Update .dockerignore

Make sure `data/` is NOT being excluded during the build. Open `.dockerignore` and remove the `data/` line:
```
# Python cache
__pycache__/
*.pyc
*.pyo

# Git
.git/
.gitignore

# Local environment
.env
venv/
.venv/

# NOT excluding data/ anymore — Docker needs to create it during build