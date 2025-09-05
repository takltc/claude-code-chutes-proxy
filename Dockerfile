# syntax=docker/dockerfile:1
# Minimal image to run the FastAPI proxy with Uvicorn

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install runtime tools for healthcheck
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first for better layer caching
COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy source
COPY app ./app
COPY run.sh ./run.sh

# Default environment (can be overridden at runtime)
ENV PORT=8080 \
    UVICORN_WORKERS=1 \
    CHUTES_BASE_URL=https://llm.chutes.ai

# Expose the internal port (defaults to 8080)
EXPOSE 8080

# Healthcheck the root endpoint
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -fsS "http://localhost:${PORT}/" || exit 1

# Run via the provided launcher script
CMD ["./run.sh"]

