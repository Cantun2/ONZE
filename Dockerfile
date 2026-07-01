# ONZE backend image: FastAPI served by uvicorn.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install deps first for layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# App code + central config.
COPY config.yaml ./
COPY src ./src

EXPOSE 8000

# ONZE_ODDS_API_KEY is injected at runtime via env (never baked into the image).
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
