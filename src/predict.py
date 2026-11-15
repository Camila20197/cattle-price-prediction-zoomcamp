"""Web service: will a lot of steers sell above the typical price?

Loads the model trained by src/train.py and serves it with Flask, in two ways:

    Web pages (in Spanish, for producers)
        GET  /           form to describe a lot
        POST /           shows the result for that lot
        GET  /acerca     how the model works and its limits

    JSON API (in English)
        GET  /health     -> {"status": "ok"}
        POST /predict    -> receives one lot as JSON, returns the premium probability

Run locally (from the src folder, so Python finds predict.py):
    cd src
    uv run waitress-serve --listen=0.0.0.0:9696 predict:app
"""
from __future__ import annotations

import json
import math
import os
import pickle
from datetime import date
from pathlib import Path

from flask import Flask, jsonify, render_template, request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = Path(os.getenv("MODEL_PATH", PROJECT_ROOT / "models" / "model.bin"))
OPTIONS_FILE = Path(os.getenv("OPTIONS_PATH", PROJECT_ROOT / "models" / "form_options.json"))

# Same limits as the cleaning rules in train.py
MIN_HEAD, MAX_HEAD = 1, 1_000
MIN_WEIGHT_KG, MAX_WEIGHT_KG = 100, 1_100
PROVINCES = ["Entre Rios", "Santa Fe", "Corrientes", "Misiones"]

with MODEL_FILE.open("rb") as file:
    dv, model = pickle.load(file)
FORM_OPTIONS = json.loads(OPTIONS_FILE.read_text(encoding="utf-8"))

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
REQUIRED_FIELDS = STRICT_FIELDS + LENIENT_FIELDS + list(NUMERIC_FIELDS) + ["date"]

app = Flask(__name__)  # __name__ lets Flask find templates/ and static/ next to this file


# ---------------------------------------------------------------------------
# Model logic, shared by the web pages and the JSON API
# ---------------------------------------------------------------------------
def validate(lot: dict) -> tuple[dict[str, str], dict[str, str]]:
    """Return (problems, warnings), each as {field: message in English}."""
    problems, warnings = {}, {}
    missing = [field for field in REQUIRED_FIELDS if field not in lot]
    if missing:
        return {"missing": f"missing fields: {', '.join(missing)}"}, warnings

    for field in STRICT_FIELDS:
        if lot[field] not in KNOWN_VALUES[field]:
            problems[field] = f"unknown {field} '{lot[field]}'. Valid values: {sorted(KNOWN_VALUES[field])}"
    if lot["origin_province"] not in PROVINCES:
        problems["origin_province"] = f"the model only covers lots from {', '.join(PROVINCES)}"

    for field in LENIENT_FIELDS:
        if lot[field] not in KNOWN_VALUES[field]:
            warnings[field] = f"{field} '{lot[field]}' was not seen in training; the prediction is less reliable"

    for field, (low, high) in NUMERIC_FIELDS.items():
        value = lot[field]
        if not isinstance(value, (int, float)) or not low <= value <= high:
            problems[field] = f"{field} must be a number between {low} and {high}"

    try:
        date.fromisoformat(lot["date"])
    except (TypeError, ValueError):
        problems["date"] = "date must be written as YYYY-MM-DD"
    return problems, warnings


def to_features(lot: dict) -> dict:
    """Turn a lot into the features used in training (same as notebook section 6.1)."""
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


# ---------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    lot = request.get_json(silent=True)
    if not isinstance(lot, dict):
        return jsonify({"errors": ["send one lot as a JSON object"]}), 400

    problems, warnings = validate(lot)
    if problems:
        return jsonify({"errors": list(problems.values())}), 400

    probability = predict_premium(lot)
    return jsonify({
        "premium_probability": round(probability, 3),
        "premium": probability >= 0.5,
        "warnings": list(warnings.values()),
    })


# ---------------------------------------------------------------------------
# Web pages (Spanish). They only translate between the form and the model.
# ---------------------------------------------------------------------------
ZONE_NAMES = {
    "ZONA 1": "Gran Buenos Aires",
    "ZONA 2": "Norte y periferia de Buenos Aires",
    "ZONA 3": "Centro de Buenos Aires",
    "ZONA 4": "Sur de Buenos Aires",
    "ZONA 5": "Córdoba, Santa Fe y Entre Ríos",
    "ZONA 6": "San Luis y La Pampa",
    "ZONA 7": "Patagonia",
    "ZONA 8": "Cuyo",
    "ZONA 9": "Noreste",
    "ZONA 10": "Noroeste",
}
PROVINCE_NAMES = {"Entre Rios": "Entre Ríos", "Santa Fe": "Santa Fe", "Corrientes": "Corrientes", "Misiones": "Misiones"}
FIELD_MESSAGES = {
    "missing": "Faltan datos del lote.",
    "category": "Elegí una categoría de la lista.",
    "origin_province": "El modelo solo cubre lotes de Entre Ríos, Santa Fe, Corrientes y Misiones.",
    "dest_zone": "Elegí una zona de destino de la lista.",
    "head": f"La cantidad de cabezas tiene que estar entre {MIN_HEAD} y {MAX_HEAD:,}.".replace(",", "."),
    "avg_weight": f"El peso promedio tiene que estar entre {MIN_WEIGHT_KG} y {MAX_WEIGHT_KG:,} kg.".replace(",", "."),
    "date": "Escribí una fecha válida.",
    "breed": "Esa raza no aparece en los datos de entrenamiento: tomá el resultado con más cuidado.",
    "origin_department": "Ese departamento no aparece en los datos de entrenamiento: tomá el resultado con más cuidado.",
}


FEATURE_NAMES = {
    "log_weight": "Peso promedio",
    "origin_department": "Departamento de origen",
    "log_head": "Cantidad de cabezas",
    "category": "Categoría",
    "weekday": "Día de la semana",
    "breed": "Raza",
    "dest_zone": "Zona de destino",
    "origin_province": "Provincia",
}


def feature_importance() -> list[tuple[str, int]]:
    """Importance of each variable in the model, as % (one-hot columns added up per variable)."""
    totals: dict[str, float] = {}
    for name, value in zip(dv.get_feature_names_out(), model.feature_importances_):
        variable = name.split("=")[0]
        totals[variable] = totals.get(variable, 0) + value
    ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    return [(FEATURE_NAMES[variable], round(value * 100)) for variable, value in ranked]


IMPORTANCE = feature_importance()


def category_label(category: str) -> str:
    """'Bovino Novillitos Regulares' -> 'Novillitos Regulares'."""
    return category.removeprefix("Bovino ")


def breed_label(breed: str) -> str:
    return {"Otra 99": "Otra / sin dato", "Bovino Criollo": "Criollo", "Limuosin": "Limousin"}.get(breed, breed)


def default_form() -> dict:
    return {
        "category": "Bovino Novillitos Regulares",
        "breed": "Aberdeen Angus",
        "origin_province": "Entre Rios",
        "origin_department": "",
        "dest_zone": "ZONA 5",
        "date": date.today().isoformat(),
        "head": "",
        "avg_weight": "",
    }


def lot_from_form(form) -> dict:
    """Read the form; numbers that cannot be read are left as text so validate() rejects them."""
    lot = {field: form.get(field, "").strip() for field in REQUIRED_FIELDS}
    for field in NUMERIC_FIELDS:
        try:
            lot[field] = float(lot[field].replace(",", "."))
        except ValueError:
            pass
    return lot


def render_form(data: dict, result: dict | None = None, errors: list[str] | None = None, status: int = 200):
    page = render_template(
        "index.html",
        data=data, result=result, errors=errors or [],
        options=FORM_OPTIONS, zone_names=ZONE_NAMES, province_names=PROVINCE_NAMES, importance=IMPORTANCE,
        category_label=category_label, breed_label=breed_label,
        limits={"head": (MIN_HEAD, MAX_HEAD), "avg_weight": (MIN_WEIGHT_KG, MAX_WEIGHT_KG)},
    )
    return page, status


@app.get("/")
def index():
    return render_form(default_form())


@app.post("/")
def evaluate():
    data = request.form.to_dict()
    lot = lot_from_form(request.form)
    problems, warnings = validate(lot)
    if problems:
        return render_form(data, errors=[FIELD_MESSAGES[field] for field in problems], status=400)

    probability = predict_premium(lot)
    result = {
        "percent": round(probability * 100),
        "premium": probability >= 0.5,
        "warnings": [FIELD_MESSAGES[field] for field in warnings],
        "summary": (f"{lot['head']:g} cabezas de {category_label(lot['category']).lower()}, "
                    f"{lot['avg_weight']:g} kg promedio, desde {lot['origin_department']} "
                    f"({PROVINCE_NAMES[lot['origin_province']]}) hacia la {lot['dest_zone'].title()}"),
    }
    return render_form(data, result=result)


@app.get("/acerca")
def about():
    return render_template("acerca.html")


if __name__ == "__main__":
    # Development server only; use waitress-serve in production (see the module docstring)
    app.run(debug=True, host="0.0.0.0", port=9696)
