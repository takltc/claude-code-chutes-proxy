#!/usr/bin/env sh
set -euo pipefail

# Defaults (can be overridden by env)
: "${PORT:=8080}"
: "${UVICORN_WORKERS:=1}"

echo "Starting FastAPI proxy on :${PORT} with ${UVICORN_WORKERS} worker(s)"
exec uvicorn app.main:app \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --workers "${UVICORN_WORKERS}"

