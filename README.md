# AutoEstim v3 — Car Price Prediction

> Dataset réel Craigslist USA · Optuna Tuning · Cross-Validation 5-folds · Feature Importance · Docker

## Nouveautés v3

| Amélioration | Fichier modifié |
|---|---|
| **Hyperparameter Tuning avec Optuna** | `train_models.py` |
| **Cross-Validation 5 folds** | `train_models.py` + `main.py` + `app.js` |
| **Feature Importance** | `train_models.py` + `main.py` + `app.js` + `index.html` |
| **Docker + docker-compose** | `Dockerfile` (×2) + `docker-compose.yml` |

## Démarrage rapide

### Option A — Sans Docker (développement)

```bash
cd backend
py -m venv venv && venv\Scripts\activate     # Windows
pip install -r requirements.txt

# Préparer les données (une seule fois)
py prepare_data.py --input vehicles.csv

# Entraîner avec Optuna (recommandé, ~45 min)
py train_models.py

# Entraîner sans Optuna (rapide, ~15 min)
py train_models.py --no-tune

# Lancer l'API
py -m uvicorn main:app --reload --port 8000

# Dans un autre terminal
cd frontend && py -m http.server 5500 --bind 127.0.0.1
```

### Option B — Avec Docker (production)

```bash
# Pré-requis : avoir généré les modèles (étapes prepare_data + train_models)

docker-compose up --build

# Frontend : http://localhost:5500
# API      : http://localhost:8000
# Docs API : http://localhost:8000/docs
```

## Options de train_models.py

```bash
py train_models.py               # Optuna 30 trials (recommandé)
py train_models.py --no-tune     # Params par défaut (rapide)
py train_models.py --trials 15   # Optuna 15 trials (compromis)
```

## Nouveaux endpoints API v3

| Endpoint | Description |
|---|---|
| `GET /api/feature-importance/{model}` | Top features qui influencent le prix |
| `GET /api/cross-validation` | Résultats CV 5-folds des deux modèles |

## Pipeline ML complet

```
vehicles.csv (1.35GB)
    ↓ prepare_data.py
vehicles_clean.csv (353k lignes propres)
    ↓ Optuna (30 trials, CV 3-folds)
Meilleurs hyperparamètres RF + XGBoost
    ↓ train_models.py
Cross-Validation 5-folds → métriques robustes
    ↓ Entraînement final
random_forest_pipeline.joblib + xgboost_pipeline.joblib
Feature Importance extraite → metrics.json
    ↓ main.py (FastAPI)
API REST → frontend
```

## Stack technique

- **ML** : scikit-learn, XGBoost, Optuna, pandas, numpy
- **Backend** : FastAPI, Uvicorn, SQLite
- **Frontend** : HTML / CSS / JavaScript vanilla
- **DevOps** : Docker, docker-compose, Nginx
