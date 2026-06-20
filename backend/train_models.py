"""
train_models.py — v3
====================
Améliorations vs v2 :
  1. Cross-Validation (5 folds) pour une évaluation plus robuste
  2. Hyperparameter Tuning avec Optuna (remplace les valeurs arbitraires)
  3. Feature Importance extraite et sauvegardée pour le frontend

Workflow :
  python prepare_data.py --input vehicles.csv
  python train_models.py

  Options :
    --no-tune      : saute Optuna (entraînement rapide avec params par défaut)
    --trials N     : nombre d'essais Optuna (défaut: 30)

  Exemples :
    python train_models.py                  # tuning complet (recommandé)
    python train_models.py --no-tune        # rapide, bons params par défaut
    python train_models.py --trials 15      # tuning léger
"""

import argparse
import json
import time
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

DATA_PATH = "data/vehicles_clean.csv"
MODELS_DIR = "models"

NUMERIC_FEATURES = ["year", "odometer"]
CATEGORICAL_FEATURES = [
    "manufacturer", "condition", "fuel",
    "transmission", "drive", "type", "state",
]
TARGET = "price"

MAX_CATEGORIES = {
    "manufacturer": 30, "state": 30, "type": 20,
    "condition": 10, "fuel": 10, "transmission": 5, "drive": 5,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def keep_top_categories(df, col, max_cats):
    top = df[col].value_counts().nlargest(max_cats).index
    df[col] = df[col].where(df[col].isin(top), other="other")
    return df


def build_preprocessor():
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])


def evaluate(name, y_true, y_pred, train_sec):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"\n{'='*45}")
    print(f"  {name}")
    print(f"{'='*45}")
    print(f"  MAE  : ${mae:>10,.2f}")
    print(f"  RMSE : ${rmse:>10,.2f}")
    print(f"  R²   : {r2:>12.4f}")
    print(f"  Temps: {train_sec:>11.1f}s")
    return {"mae": round(float(mae), 2), "rmse": round(float(rmse), 2),
            "r2": round(float(r2), 4), "train_seconds": round(train_sec, 1)}


# ── Cross-Validation ──────────────────────────────────────────────────────────

def cross_validate_pipeline(pipeline, X, y, n_folds=5):
    """
    Évalue le pipeline avec une validation croisée à n_folds folds.
    Retourne MAE, RMSE et R² moyens ± écart-type sur les n_folds folds.
    """
    print(f"\n  Cross-Validation ({n_folds} folds) en cours...")
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    r2_scores   = cross_val_score(pipeline, X, y, cv=kf, scoring="r2",
                                  n_jobs=-1)
    mae_scores  = -cross_val_score(pipeline, X, y, cv=kf,
                                   scoring="neg_mean_absolute_error", n_jobs=-1)
    rmse_scores = np.sqrt(-cross_val_score(pipeline, X, y, cv=kf,
                                           scoring="neg_mean_squared_error",
                                           n_jobs=-1))

    cv_results = {
        "n_folds": n_folds,
        "r2_mean":   round(float(r2_scores.mean()),   4),
        "r2_std":    round(float(r2_scores.std()),    4),
        "mae_mean":  round(float(mae_scores.mean()),  2),
        "mae_std":   round(float(mae_scores.std()),   2),
        "rmse_mean": round(float(rmse_scores.mean()), 2),
        "rmse_std":  round(float(rmse_scores.std()),  2),
    }

    print(f"  R²   : {cv_results['r2_mean']:.4f} ± {cv_results['r2_std']:.4f}")
    print(f"  MAE  : ${cv_results['mae_mean']:,.2f} ± ${cv_results['mae_std']:,.2f}")
    print(f"  RMSE : ${cv_results['rmse_mean']:,.2f} ± ${cv_results['rmse_std']:,.2f}")
    return cv_results


# ── Feature Importance ────────────────────────────────────────────────────────

def extract_feature_importance(pipeline, top_n=15):
    """
    Extrait les feature importances du modèle entraîné et les mappe
    vers les noms de features lisibles (après OneHotEncoding).
    Retourne une liste triée des top_n features les plus importantes.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    regressor    = pipeline.named_steps["regressor"]

    # Noms des features numériques
    num_names = NUMERIC_FEATURES

    # Noms des features catégorielles après OneHotEncoding
    ohe = preprocessor.named_transformers_["cat"]
    cat_names = []
    for col, cats in zip(CATEGORICAL_FEATURES, ohe.categories_):
        for cat in cats:
            cat_names.append(f"{col}={cat}")

    all_feature_names = num_names + cat_names

    # Importances selon le type de modèle
    if hasattr(regressor, "feature_importances_"):
        importances = regressor.feature_importances_
    else:
        return []

    # Tri et sélection des top_n
    indices = np.argsort(importances)[::-1][:top_n]
    result = []
    for rank, idx in enumerate(indices):
        if idx < len(all_feature_names):
            result.append({
                "rank":       rank + 1,
                "feature":    all_feature_names[idx],
                "importance": round(float(importances[idx]), 6),
                "importance_pct": round(float(importances[idx]) * 100, 2),
            })
    return result


# ── Optuna Tuning ─────────────────────────────────────────────────────────────

def tune_random_forest(X_train, y_train, n_trials=5):
    """
    Recherche les meilleurs hyperparamètres pour RandomForestRegressor via Optuna.
    Objectif : maximiser le R² en validation croisée 3-folds.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  Optuna non installé — params par défaut utilisés.")
        return {"n_estimators": 10, "max_depth": 16, "min_samples_split": 5,
                "min_samples_leaf": 2, "max_features": "sqrt"}

    print(f"\n  Optuna — Random Forest ({n_trials} trials)...")

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators",10,20),
            "max_depth":        trial.suggest_int("max_depth", 8, 20),
            "min_samples_split":trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features":     trial.suggest_categorical("max_features",
                                                          ["sqrt", "log2", 0.5]),
        }
        pipe = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("regressor", RandomForestRegressor(**params, random_state=42, n_jobs=-1)),
        ])
        scores = cross_val_score(pipe, X_train, y_train, cv=3,
                                 scoring="r2", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print(f"  Meilleurs params RF : {best}")
    print(f"  Meilleur R² CV     : {study.best_value:.4f}")
    return best


def tune_xgboost(X_train, y_train, n_trials=5):
    """
    Recherche les meilleurs hyperparamètres pour XGBRegressor via Optuna.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  Optuna non installé — params par défaut utilisés.")
        return {"n_estimators": 20, "max_depth": 7, "learning_rate": 0.05,
                "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5}

    print(f"\n  Optuna — XGBoost ({n_trials} trials)...")

    def objective(trial):
        params = {
            "n_estimators":    trial.suggest_int("n_estimators", 20, 30),
            "max_depth":       trial.suggest_int("max_depth", 4, 10),
            "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample":       trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight":trial.suggest_int("min_child_weight", 1, 10),
            "gamma":           trial.suggest_float("gamma", 0, 0.5),
        }
        pipe = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("regressor", XGBRegressor(**params, random_state=42, n_jobs=-1,
                                       verbosity=0)),
        ])
        scores = cross_val_score(pipe, X_train, y_train, cv=3,
                                 scoring="r2", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print(f"  Meilleurs params XGB : {best}")
    print(f"  Meilleur R² CV      : {study.best_value:.4f}")
    return best


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-tune", action="store_true",
                        help="Saute Optuna, utilise les params par défaut")
    parser.add_argument("--trials", type=int, default=30,
                        help="Nombre d'essais Optuna (défaut: 30)")
    args = parser.parse_args()

    # ── Chargement ──────────────────────────────────────────────────────────
    print(f"Chargement de {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"  → {len(df):,} lignes")

    for col, max_cats in MAX_CATEGORIES.items():
        if col in df.columns:
            df = keep_top_categories(df, col, max_cats)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train : {len(X_train):,} | Test : {len(X_test):,}")

    metrics  = {}
    cv_data  = {}
    fi_data  = {}

    # ── Random Forest ────────────────────────────────────────────────────────
    print("\n" + "─"*50)
    print("  RANDOM FOREST")
    print("─"*50)

    if args.no_tune:
        rf_params = {"n_estimators": 20, "max_depth": 16,
                     "min_samples_split": 5, "min_samples_leaf": 2,
                     "max_features": "sqrt"}
        print("  Params par défaut (--no-tune)")
    else:
        rf_params = tune_random_forest(X_train, y_train, n_trials=args.trials)

    rf_pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("regressor", RandomForestRegressor(**rf_params, random_state=42, n_jobs=-1)),
    ])

    # Cross-Validation
    cv_data["random_forest"] = cross_validate_pipeline(rf_pipeline, X_train, y_train)

    # Entraînement final sur tout le train set
    print("\n  Entraînement final Random Forest...")
    start = time.time()
    rf_pipeline.fit(X_train, y_train)
    rf_time = time.time() - start

    metrics["random_forest"] = evaluate(
        "Random Forest Regressor", y_test,
        rf_pipeline.predict(X_test), rf_time
    )
    metrics["random_forest"]["best_params"] = rf_params
    fi_data["random_forest"] = extract_feature_importance(rf_pipeline)

    joblib.dump(rf_pipeline, f"{MODELS_DIR}/random_forest_pipeline.joblib")
    print("  ✓ random_forest_pipeline.joblib sauvegardé")

    # ── XGBoost ──────────────────────────────────────────────────────────────
    print("\n" + "─"*50)
    print("  XGBOOST")
    print("─"*50)

    if args.no_tune:
        xgb_params = {"n_estimators": 20, "max_depth": 7, "learning_rate": 0.05,
                      "subsample": 0.8, "colsample_bytree": 0.8,
                      "min_child_weight": 5, "gamma": 0}
        print("  Params par défaut (--no-tune)")
    else:
        xgb_params = tune_xgboost(X_train, y_train, n_trials=args.trials)

    xgb_pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("regressor", XGBRegressor(**xgb_params, random_state=42,
                                   n_jobs=-1, verbosity=0)),
    ])

    # Cross-Validation
    cv_data["xgboost"] = cross_validate_pipeline(xgb_pipeline, X_train, y_train)

    # Entraînement final
    print("\n  Entraînement final XGBoost...")
    start = time.time()
    xgb_pipeline.fit(X_train, y_train)
    xgb_time = time.time() - start

    metrics["xgboost"] = evaluate(
        "XGBoost Regressor", y_test,
        xgb_pipeline.predict(X_test), xgb_time
    )
    metrics["xgboost"]["best_params"] = xgb_params
    fi_data["xgboost"] = extract_feature_importance(xgb_pipeline)

    joblib.dump(xgb_pipeline, f"{MODELS_DIR}/xgboost_pipeline.joblib")
    print("  ✓ xgboost_pipeline.joblib sauvegardé")

    # ── Métadonnées dataset ───────────────────────────────────────────────────
    metadata = {
        "manufacturers": sorted(df["manufacturer"].unique().tolist()),
        "conditions":    sorted(df["condition"].unique().tolist()),
        "fuel_types":    sorted(df["fuel"].unique().tolist()),
        "transmissions": sorted(df["transmission"].unique().tolist()),
        "drives":        sorted(df["drive"].unique().tolist()),
        "types":         sorted(df["type"].unique().tolist()),
        "states":        sorted(df["state"].unique().tolist()),
        "year_min":      int(df["year"].min()),
        "year_max":      int(df["year"].max()),
        "odometer_avg":  int(df["odometer"].mean()),
        "price_min":     float(df["price"].min()),
        "price_max":     float(df["price"].max()),
        "price_mean":    float(df["price"].mean()),
        "dataset_size":  len(df),
        "currency":      "USD",
    }

    # ── Sauvegarde metrics.json ───────────────────────────────────────────────
    with open(f"{MODELS_DIR}/metrics.json", "w") as f:
        json.dump({
            "metrics":          metrics,
            "cross_validation": cv_data,
            "feature_importance": fi_data,
            "metadata":         metadata,
        }, f, indent=2)

    print(f"\n  ✓ metrics.json sauvegardé")

    # ── Résumé final ──────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("  RÉSUMÉ FINAL")
    print("="*50)
    print(f"  Dataset    : {len(df):,} voitures Craigslist USA")
    print(f"  Prix moyen : ${df['price'].mean():,.0f}")

    for name, label in [("random_forest","Random Forest"), ("xgboost","XGBoost")]:
        m  = metrics[name]
        cv = cv_data[name]
        fi = fi_data[name]
        print(f"\n  [{label}]")
        print(f"    Test  — MAE: ${m['mae']:,.0f} | R²: {m['r2']:.4f}")
        print(f"    CV 5  — R²: {cv['r2_mean']:.4f} ± {cv['r2_std']:.4f}")
        if fi:
            print(f"    Top feature: {fi[0]['feature']} ({fi[0]['importance_pct']:.1f}%)")
    print()


if __name__ == "__main__":
    main()
