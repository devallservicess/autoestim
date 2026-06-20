"""
API FastAPI v3 — Car Price Prediction (Craigslist USA)

Nouveaux endpoints v3 :
  GET /api/feature-importance/{model}  → top features qui influencent le prix
  GET /api/cross-validation            → résultats CV 5-folds des deux modèles
"""

import json
import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import database

app = FastAPI(title="Car Price Prediction API", version="3.0.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

MODELS_DIR = "models"
models = {}
metrics_data = {}
cv_data = {}
fi_data = {}
metadata = {}


@app.on_event("startup")
def load_resources():
    database.init_db()
    rf_path  = os.path.join(MODELS_DIR, "random_forest_pipeline.joblib")
    xgb_path = os.path.join(MODELS_DIR, "xgboost_pipeline.joblib")
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")

    if not (os.path.exists(rf_path) and os.path.exists(xgb_path)):
        raise RuntimeError("Modèles introuvables. Lance prepare_data.py puis train_models.py.")

    models["random_forest"] = joblib.load(rf_path)
    models["xgboost"]       = joblib.load(xgb_path)

    with open(metrics_path) as f:
        data = json.load(f)
        metrics_data.update(data.get("metrics", {}))
        cv_data.update(data.get("cross_validation", {}))
        fi_data.update(data.get("feature_importance", {}))
        metadata.update(data.get("metadata", {}))

    print(f"API v3 prete | {metadata.get('dataset_size', '?'):,} voitures Craigslist")


class CarInput(BaseModel):
    manufacturer: str
    year:         int  = Field(..., ge=1990, le=2026)
    odometer:     int  = Field(..., ge=0, le=500000)
    condition:    str  = Field(default="good")
    fuel:         str
    transmission: str
    drive:        str  = Field(default="unknown")
    type:         str  = Field(default="unknown")
    state:        str  = Field(default="unknown")
    model:        str  = Field(default="xgboost")


class CarPrediction(BaseModel):
    manufacturer: str
    year: int
    odometer: int
    condition: str
    fuel: str
    transmission: str
    drive: str
    type: str
    state: str
    model_used: str
    predicted_price: float


@app.get("/")
def root():
    return {"status": "ok", "version": "3.0.0"}

@app.get("/api/metadata")
def get_metadata():
    return metadata

@app.get("/api/metrics")
def get_metrics():
    return metrics_data

@app.get("/api/cross-validation")
def get_cross_validation():
    if not cv_data:
        raise HTTPException(status_code=404, detail="Données CV non disponibles.")
    return cv_data

@app.get("/api/feature-importance/{model_name}")
def get_feature_importance(model_name: str, top: int = 15):
    if model_name not in ("xgboost", "random_forest"):
        raise HTTPException(status_code=400, detail="model_name: 'xgboost' ou 'random_forest'")
    if model_name not in fi_data:
        raise HTTPException(status_code=404, detail="Feature importance non disponible.")
    return {"model": model_name, "features": fi_data[model_name][:top]}

@app.post("/api/predict", response_model=CarPrediction)
def predict_price(car: CarInput):
    if car.model not in models:
        raise HTTPException(status_code=400, detail="Modele invalide.")
    input_df = pd.DataFrame([{
        "year": car.year, "odometer": car.odometer,
        "manufacturer": car.manufacturer.lower().strip(),
        "condition":    car.condition.lower().strip(),
        "fuel":         car.fuel.lower().strip(),
        "transmission": car.transmission.lower().strip(),
        "drive":        car.drive.lower().strip(),
        "type":         car.type.lower().strip(),
        "state":        car.state.lower().strip(),
    }])
    predicted_price = float(models[car.model].predict(input_df)[0])
    predicted_price = max(round(predicted_price, 2), 0)
    record = {
        "manufacturer": car.manufacturer, "year": car.year,
        "odometer": car.odometer, "condition": car.condition,
        "fuel": car.fuel, "transmission": car.transmission,
        "drive": car.drive, "type": car.type, "state": car.state,
        "model_used": car.model, "predicted_price": predicted_price,
    }
    database.insert_car(record)
    return record

@app.get("/api/cars")
def list_cars():
    return database.get_all_cars()

@app.delete("/api/cars/{car_id}")
def remove_car(car_id: int):
    if not database.delete_car(car_id):
        raise HTTPException(status_code=404, detail="Voiture non trouvee")
    return {"status": "deleted", "id": car_id}

@app.get("/api/dashboard")
def dashboard_stats():
    return database.get_dashboard_stats()
