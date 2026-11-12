"""Web service: will a lot of steers sell above the typical price?

Loads the model trained by src/train.py and serves it with Flask.

Endpoints:
    GET  /health    -> {"status": "ok"}
    POST /predict   -> receives one lot as JSON, returns the premium probability

Run locally (from the src folder, so Python finds predict.py):
    cd src
    uv run waitress-serve --listen=0.0.0.0:9696 predict:app
"""
from __future__ import annotations

import math
import os
import pickle
from datetime import date
from pathlib import Path

from flask import Flask, jsonify, request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = Path(os.getenv("MODEL_PATH", PROJECT_ROOT / "models" / "model.bin"))

# Same limits as the cleaning rules in train.py
MIN_HEAD, MAX_HEAD = 1, 1_000
MIN_WEIGHT_KG, MAX_WEIGHT_KG = 100, 1_100
PROVINCES = ["Entre Rios", "Santa Fe", "Corrientes", "Misiones"]

with MODEL_FILE.open("rb") as file:
    dv, model = pickle.load(file)

# Values the model saw during training, read from the DictVectorizer:
# {"category": {"Bovino Novillitos Regulares", ...}, "breed": {...}, ...}
KNOWN_VALUES: dict[str, set[str]] = {}
for name in dv.get_feature_names_out():
    if "=" in name:
        feature, value = name.split("=", 1)
        KNOWN_VALUES.setdefault(feature, set()).add(value)

# Must match one of the training values, otherwise the request is rejected
STRICT_FIELDS = ["category", "origin_province", "dest_zone"]
# May be new (the model then treats them as "none of the known values"), but a warning is returned
LENIENT_FIELDS = ["breed", "origin_department"]
NUMERIC_FIELDS = {"head": (MIN_HEAD, MAX_HEAD), "avg_weight": (MIN_WEIGHT_KG, MAX_WEIGHT_KG)}

app = Flask("cattle-price")


def validate(lot: dict) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for one lot."""
    errors, warnings = [], []
    required = STRICT_FIELDS + LENIENT_FIELDS + list(NUMERIC_FIELDS) + ["date"]
    missing = [field for field in required if field not in lot]
    if missing:
        return [f"missing fields: {', '.join(missing)}"], warnings

    for field in STRICT_FIELDS:
        if lot[field] not in KNOWN_VALUES[field]:
            errors.append(f"unknown {field} '{lot[field]}'. Valid values: {sorted(KNOWN_VALUES[field])}")
    if lot["origin_province"] not in PROVINCES:
        errors.append(f"the model only covers lots from {', '.join(PROVINCES)}")

    for field in LENIENT_FIELDS:
        if lot[field] not in KNOWN_VALUES[field]:
            warnings.append(f"{field} '{lot[field]}' was not seen in training; the prediction is less reliable")

    for field, (low, high) in NUMERIC_FIELDS.items():
        value = lot[field]
        if not isinstance(value, (int, float)) or not low <= value <= high:
            errors.append(f"{field} must be a number between {low} and {high}")

    try:
        date.fromisoformat(lot["date"])
    except (TypeError, ValueError):
        errors.append("date must be written as YYYY-MM-DD")
    return errors, warnings


def to_features(lot: dict) -> dict:
    """Turn the request into the features used in training (same as notebook section 6.1)."""
    return {
        "category": lot["category"],
        "breed": lot["breed"],
        "origin_province": lot["origin_province"],
        "origin_department": lot["origin_department"],
        "dest_zone": lot["dest_zone"],
        "weekday": date.fromisoformat(lot["date"]).strftime("%A"),
        "log_head": math.log(lot["head"]),
        "log_weight": math.log(lot["avg_weight"]),
    }


def predict_premium(lot: dict) -> float:
    X = dv.transform([to_features(lot)])
    return float(model.predict_proba(X)[0, 1])


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    lot = request.get_json(silent=True)
    if not isinstance(lot, dict):
        return jsonify({"errors": ["send one lot as a JSON object"]}), 400

    errors, warnings = validate(lot)
    if errors:
        return jsonify({"errors": errors}), 400

    probability = predict_premium(lot)
    return jsonify({
        "premium_probability": round(probability, 3),
        "premium": probability >= 0.5,
        "warnings": warnings,
    })


if __name__ == "__main__":
    # Development server only; use waitress-serve in production (see the module docstring)
    app.run(debug=True, host="0.0.0.0", port=9696)
