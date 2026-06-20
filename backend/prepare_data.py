"""
Nettoyage et préparation du dataset Craigslist vehicles.csv pour
l'entraînement des modèles de prédiction de prix.

Usage :
    python prepare_data.py --input vehicles.csv
    python prepare_data.py --input vehicles.csv --sample 50000

Le script :
1. Charge le CSV brut (26 colonnes, ~400k lignes)
2. Sélectionne les colonnes pertinentes pour la prédiction
3. Nettoie les valeurs aberrantes (prix <500$ ou >150000$, années,
   kilométrages impossibles)
4. Supprime les lignes avec trop de valeurs manquantes
5. Rempli les valeurs manquantes restantes par stratégie par colonne
6. Sauvegarde le dataset nettoyé dans data/vehicles_clean.csv
7. Génère un résumé statistique du dataset nettoyé
"""

import argparse
import sys

import numpy as np
import pandas as pd

CURRENT_YEAR = 2026

# Colonnes sélectionnées depuis les 26 disponibles dans le CSV Craigslist
SELECTED_COLS = [
    "price",        # cible (y)
    "year",         # numérique
    "odometer",     # numérique (kilométrage en miles)
    "manufacturer", # catégorielle
    "condition",    # catégorielle
    "fuel",         # catégorielle
    "transmission", # catégorielle
    "drive",        # catégorielle (fwd/rwd/4wd)
    "type",         # catégorielle (sedan/suv/truck/...)
    "state",        # catégorielle (état US)
]

# Valeurs aberrantes à filtrer
PRICE_MIN = 500
PRICE_MAX = 150_000
YEAR_MIN = 1990
ODOMETER_MAX = 500_000


def load_and_clean(input_path: str, sample: int | None = None) -> pd.DataFrame:
    print(f"Chargement de {input_path}...")
    df = pd.read_csv(
        input_path,
        usecols=lambda c: c in SELECTED_COLS,
        low_memory=False,
    )
    print(f"  → {len(df):,} lignes chargées, {len(df.columns)} colonnes")

    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=42).reset_index(drop=True)
        print(f"  → Échantillon aléatoire : {len(df):,} lignes")

    # --- Filtrage des prix aberrants ---
    before = len(df)
    df = df[(df["price"] >= PRICE_MIN) & (df["price"] <= PRICE_MAX)]
    print(f"Filtrage prix [{PRICE_MIN}$–{PRICE_MAX}$] : {before - len(df):,} lignes supprimées")

    # --- Filtrage des années impossibles ---
    before = len(df)
    df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= CURRENT_YEAR)]
    print(f"Filtrage années [{YEAR_MIN}–{CURRENT_YEAR}] : {before - len(df):,} lignes supprimées")

    # --- Filtrage du kilométrage ---
    before = len(df)
    df = df[df["odometer"] <= ODOMETER_MAX]
    df = df[df["odometer"] >= 0]
    print(f"Filtrage odomètre [0–{ODOMETER_MAX:,}] : {before - len(df):,} lignes supprimées")

    # --- Nettoyage des catégorielles ---
    # Uniformiser la casse
    cat_cols = ["manufacturer", "condition", "fuel", "transmission", "drive", "type", "state"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()
            df[col] = df[col].replace("nan", np.nan)

    # Garder seulement les valeurs connues pour transmission et fuel
    valid_transmissions = {"automatic", "manual", "other"}
    valid_fuels = {"gas", "diesel", "hybrid", "electric", "other"}

    df["transmission"] = df["transmission"].where(df["transmission"].isin(valid_transmissions), other=np.nan)
    df["fuel"] = df["fuel"].where(df["fuel"].isin(valid_fuels), other=np.nan)

    # --- Suppression lignes avec trop de NaN sur colonnes critiques ---
    critical_cols = ["price", "year", "odometer", "manufacturer", "fuel", "transmission"]
    before = len(df)
    df = df.dropna(subset=critical_cols)
    print(f"Suppression NaN critiques : {before - len(df):,} lignes supprimées")

    # --- Remplissage des NaN restants (colonnes non critiques) ---
    fill_strategies = {
        "condition": "good",
        "drive": "unknown",
        "type": "unknown",
        "state": "unknown",
    }
    for col, fill_val in fill_strategies.items():
        if col in df.columns:
            df[col] = df[col].fillna(fill_val)

    # --- Conversion des types ---
    df["year"] = df["year"].astype(int)
    df["odometer"] = df["odometer"].astype(int)
    df["price"] = df["price"].astype(float)

    print(f"\nDataset final : {len(df):,} lignes propres")
    return df


def main():
    parser = argparse.ArgumentParser(description="Prépare le dataset Craigslist vehicles.csv")
    parser.add_argument("--input", default="vehicles.csv", help="Chemin vers le CSV brut")
    parser.add_argument("--sample", type=int, default=None, help="Limite le nombre de lignes (optionnel)")
    args = parser.parse_args()

    df = load_and_clean(args.input, args.sample)

    output_path = "data/vehicles_clean.csv"
    df.to_csv(output_path, index=False)
    print(f"\nDataset nettoyé sauvegardé → {output_path}")

    print("\n=== Statistiques du prix (USD) ===")
    print(df["price"].describe().apply(lambda x: f"${x:,.2f}"))

    print("\n=== Répartition carburant ===")
    print(df["fuel"].value_counts())

    print("\n=== Répartition transmission ===")
    print(df["transmission"].value_counts())

    print("\n=== Top 10 constructeurs ===")
    print(df["manufacturer"].value_counts().head(10))


if __name__ == "__main__":
    main()
