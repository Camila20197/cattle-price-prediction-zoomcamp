"""Tests for the web service. They use Flask's test client, so no server has to be running.

Run `uv run python src/train.py` first: the service needs models/model.bin.
"""
from pathlib import Path

import pytest

MODEL_FILE = Path(__file__).resolve().parents[1] / "models" / "model.bin"
if not MODEL_FILE.exists():
    pytest.skip("models/model.bin not found: run `uv run python src/train.py` first", allow_module_level=True)

from predict import app  # noqa: E402  (imported after the check above)

LOT = {
    "category": "Bovino Novillitos Regulares",
    "breed": "Braford",
    "origin_province": "Entre Rios",
    "origin_department": "Diamante",
    "dest_zone": "ZONA 5",
    "date": "2018-05-15",
    "head": 30,
    "avg_weight": 380,
}


@pytest.fixture
def client():
    return app.test_client()


def test_health(client):
    assert client.get("/health").get_json() == {"status": "ok"}


def test_valid_lot_returns_a_probability(client):
    response = client.post("/predict", json=LOT)
    body = response.get_json()
    assert response.status_code == 200
    assert 0 <= body["premium_probability"] <= 1
    assert body["premium"] == (body["premium_probability"] >= 0.5)
    assert body["warnings"] == []


def test_missing_field_is_rejected(client):
    lot = {key: value for key, value in LOT.items() if key != "head"}
    response = client.post("/predict", json=lot)
    assert response.status_code == 400
    assert "missing fields: head" in response.get_json()["errors"][0]


@pytest.mark.parametrize("field, value", [
    ("category", "Bovino Vaca Buena"),     # not a steer category
    ("origin_province", "Cordoba"),        # outside the Litoral
    ("head", 0),                           # impossible lot size
    ("avg_weight", 2_000),                 # impossible weight
    ("date", "15/05/2018"),                # wrong date format
])
def test_invalid_values_are_rejected(client, field, value):
    response = client.post("/predict", json={**LOT, field: value})
    assert response.status_code == 400


def test_unknown_breed_gives_a_warning(client):
    response = client.post("/predict", json={**LOT, "breed": "Wagyu"})
    assert response.status_code == 200
    assert "breed 'Wagyu'" in response.get_json()["warnings"][0]
