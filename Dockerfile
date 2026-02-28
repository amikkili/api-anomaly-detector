# ─────────────────────────────────────────────────────────────────────────────
# BASE IMAGE
# We use slim — smaller image size, faster deployments on Render free tier
# Python 3.11 matches what we're developing with locally
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ─────────────────────────────────────────────────────────────────────────────
# WORKING DIRECTORY inside the container
# All our files will live at /app inside the container
# ─────────────────────────────────────────────────────────────────────────────
WORKDIR /app

# ─────────────────────────────────────────────────────────────────────────────
# INSTALL DEPENDENCIES FIRST (before copying code)
# WHY THIS ORDER? Docker caches each step (layer).
# If we copy code first, every code change rebuilds dependencies from scratch.
# By installing deps first, Docker reuses the cached layer unless
# requirements.txt changes — much faster rebuilds.
# ─────────────────────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────────────────────────────────────
# COPY PROJECT FILES
# We copy serving/ and data/ separately — only what the API needs to run.
# Training scripts, raw data, notebooks etc. are NOT needed in production.
# ─────────────────────────────────────────────────────────────────────────────
COPY serving/ ./serving/

# ─────────────────────────────────────────────────────────────────────────────
# EXPOSE PORT
# Render expects your app to listen on port 8000
# This is just documentation — the actual binding happens in CMD below
# ─────────────────────────────────────────────────────────────────────────────
EXPOSE 8000

# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT VARIABLES
# Tells Python to not buffer output — logs appear instantly in Render dashboard
# ─────────────────────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONUTF8=1

# ─────────────────────────────────────────────────────────────────────────────
# STARTUP COMMAND
# host=0.0.0.0 means accept connections from outside the container (required!)
# Without 0.0.0.0, the API only accepts connections from inside the container
# workers=2 handles concurrent requests without crashing on free tier
# ─────────────────────────────────────────────────────────────────────────────
CMD ["sh", "-c", "uvicorn serving.app:app --host 0.0.0.0 --port ${PORT:-8000}"]