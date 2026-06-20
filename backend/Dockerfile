# ── AutoEstim v3 — Backend Dockerfile ──────────────────────────────────────
# Utilise Python 3.12 (stable, compatible avec toutes les libs ML)
FROM python:3.12-slim

# Métadonnées
LABEL maintainer="AutoEstim"
LABEL description="Car Price Prediction API — Craigslist USA"
LABEL version="3.0.0"

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Installer les dépendances système nécessaires pour XGBoost et pandas
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Installer les dépendances Python
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copier le code source
COPY . .

# Créer les dossiers nécessaires s'ils n'existent pas
RUN mkdir -p data models

# Exposer le port de l'API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

# Lancer l'API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
