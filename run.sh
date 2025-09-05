#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export UVICORN_WORKERS=${UVICORN_WORKERS:-1}
export CHUTES_BASE_URL=${CHUTES_BASE_URL:-http://localhost:8000}

exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}

