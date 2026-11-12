"""Train the final premium-classification model.

Reproduces the model selected in notebooks/notebook.ipynb (section 6):
a Random Forest with 100 fully grown trees that predicts whether a lot of
steers will sell above the typical price of its category that week.

The model is trained on the train + validation weeks (1-20), evaluated once
on the test weeks (21-26), and saved together with its DictVectorizer to
models/model.bin.

Usage:
    uv run python src/train.py
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_SALES = PROJECT_ROOT / "data" / "raw" / "sio-carnes-01-ene-a-30-jun-.csv"
MODEL_FILE = PROJECT_ROOT / "models" / "model.bin"

# --- Raw file format (notebook section 1) ---------------------------------
SEPARATOR, ENCODING = ";", "latin-1"
COLUMNS = {
    "provincia_origen": "origin_province",
    "departamento_origen": "origin_department",
    "zona_destino": "dest_zone",
    "fecha_comprobante": "date",
    "raza": "breed",
    "categoria": "category",
    "cabezas_comercializadas": "head",
    "Unidad de Medida": "unit",
    "precio($)_cabeza": "price_head",
    "precio($)_kg_vivo": "price_kg",
    "cantidad_de_kg": "total_kg",
}

# --- Cleaning rules justified in notebook section 1 ------------------------
SPECIES_KEPT = "Bovino"
MIN_HEAD, MAX_HEAD = 1, 1_000
MIN_WEIGHT_KG, MAX_WEIGHT_KG = 100, 1_100
KG_LIVE_UNIT = "Kg. Vivo"

# --- Project scope and targets (notebook section 2) ------------------------
PROVINCES = ["Entre Rios", "Santa Fe", "Corrientes", "Misiones"]
GROUPS = ["Novillos", "Novillitos"]
TEST_FROM_WEEK = 21          # weeks 21-26 are the held-out test period

# --- Model selected in notebook section 6 ----------------------------------
FEATURES = ["category", "breed", "origin_province", "origin_department",
            "dest_zone", "weekday", "log_head", "log_weight"]
TARGET = "premium"
MODEL_PARAMS = {"n_estimators": 100, "random_state": 1, "n_jobs": -1}


def load_clean_sales(path: Path = RAW_SALES) -> pd.DataFrame:
    """Read the raw SIO Carnes file and apply the cleaning rules of notebook section 1."""
    raw = pd.read_csv(path, sep=SEPARATOR, encoding=ENCODING)
    sales = raw[list(COLUMNS)].rename(columns=COLUMNS)
    sales["date"] = pd.to_datetime(sales["date"])

    sales = sales.drop_duplicates()
    sales = sales[sales["category"].str.split().str[0] == SPECIES_KEPT]
    sales = sales[sales["head"].between(MIN_HEAD, MAX_HEAD)]

    kg_live = sales["unit"] == KG_LIVE_UNIT
    sales = sales.assign(avg_weight=np.where(kg_live, sales["total_kg"] / sales["head"], np.nan))
    bad_weight = kg_live & ~sales["avg_weight"].between(MIN_WEIGHT_KG, MAX_WEIGHT_KG)
    return sales[~bad_weight].reset_index(drop=True)


def build_steers_dataset(sales: pd.DataFrame) -> pd.DataFrame:
    """Scope the data and build the target, as in notebook section 2.

    The typical price (median of the same category and week) is computed on the
    steer lots of the whole country; the model is trained on the Litoral lots.
    """
    lots = sales[(sales["unit"] == KG_LIVE_UNIT) & sales["price_kg"].notna()].copy()
    lots["group"] = lots["category"].str.split().str[1]

    steers = lots[lots["group"].isin(GROUPS)].copy()
    steers["week"] = steers["date"].dt.isocalendar().week.astype(int)
    steers["weekday"] = steers["date"].dt.day_name()
    steers["typical_price"] = steers.groupby(["category", "week"])["price_kg"].transform("median")
    steers[TARGET] = (steers["price_kg"] / steers["typical_price"] > 1).astype(int)

    litoral = steers[steers["origin_province"].isin(PROVINCES)].copy()
    litoral["log_head"] = np.log(litoral["head"])
    litoral["log_weight"] = np.log(litoral["avg_weight"])
    return litoral.reset_index(drop=True)


def train(lots: pd.DataFrame) -> tuple[DictVectorizer, RandomForestClassifier]:
    """Train on weeks 1-20, report the metrics on the test weeks, return (dv, model)."""
    train_part = lots[lots["week"] < TEST_FROM_WEEK]
    test_part = lots[lots["week"] >= TEST_FROM_WEEK]

    dv = DictVectorizer(sparse=True)
    X_train = dv.fit_transform(train_part[FEATURES].to_dict(orient="records"))
    model = RandomForestClassifier(**MODEL_PARAMS).fit(X_train, train_part[TARGET])

    X_test = dv.transform(test_part[FEATURES].to_dict(orient="records"))
    probability = model.predict_proba(X_test)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    y_test = test_part[TARGET]

    print(f"Trained on {len(train_part):,} lots (weeks 1-{TEST_FROM_WEEK - 1}), "
          f"tested on {len(test_part):,} lots (weeks {TEST_FROM_WEEK}-{lots['week'].max()})")
    print(f"Test ROC AUC:  {roc_auc_score(y_test, probability):.4f}")
    print(f"Test accuracy: {accuracy_score(y_test, prediction):.3f} "
          f"(baseline {max(1 - y_test.mean(), y_test.mean()):.3f})")
    print(f"Precision: {precision_score(y_test, prediction):.3f}  "
          f"Recall: {recall_score(y_test, prediction):.3f}")
    return dv, model


def main() -> None:
    print(f"Reading {RAW_SALES.name}")
    sales = load_clean_sales()
    lots = build_steers_dataset(sales)
    print(f"Modelling dataset: {len(lots):,} steer lots from {', '.join(PROVINCES)}")

    dv, model = train(lots)

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_FILE.open("wb") as file:
        pickle.dump((dv, model), file)
    print(f"Saved DictVectorizer + model to {MODEL_FILE} "
          f"({MODEL_FILE.stat().st_size / 1_000_000:.1f} MB)")


if __name__ == "__main__":
    main()
