import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

import main
from model import load_model, predict_price, save_model
from train import generate_data, train_model


@pytest.fixture(scope="module")
def trained():
    return train_model()


@pytest.fixture
def client(monkeypatch, trained):
    monkeypatch.setattr(main, "load_model", lambda *args: trained[0])
    with TestClient(main.app) as test_client:
        yield test_client


def test_training_and_reload(tmp_path, trained):
    model, metrics = trained
    assert metrics["train_samples"] == 800
    assert metrics["test_samples"] == 200
    assert 0 < metrics["mae_rupiah"] < metrics["baseline_mae_rupiah"]
    path = tmp_path / "artifacts" / "model.joblib"
    save_model(model, path)
    expected = predict_price(model, 100, 3, 5)
    assert math.isfinite(expected)
    assert predict_price(load_model(path), 100, 3, 5) == expected


def test_data_is_reproducible():
    features, target = generate_data()
    second_features, second_target = generate_data()
    np.testing.assert_array_equal(features, second_features)
    np.testing.assert_array_equal(target, second_target)
    assert features.shape == (1000, 3)
    assert np.all((features >= [20, 1, 0]) & (features <= [500, 10, 50]))


@pytest.mark.parametrize("values", [(100, 3, 5), (20, 1, 0), (500, 10, 50)])
def test_predict(client, trained, values):
    payload = dict(zip(("luas_m2", "jumlah_kamar", "usia_tahun"), values))
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    price = response.json()["prediksi_harga_rupiah"]
    assert price == predict_price(trained[0], *values)
    assert math.isfinite(price) and price > 0


@pytest.mark.parametrize("field,value", [
    ("luas_m2", 19), ("luas_m2", 501),
    ("jumlah_kamar", 0), ("jumlah_kamar", 11),
    ("usia_tahun", -1), ("usia_tahun", 51),
    ("luas_m2", "besar"), ("luas_m2", "100"),
    ("jumlah_kamar", 2.5), ("jumlah_kamar", True),
    ("usia_tahun", None), ("usia_tahun", "5"),
])
def test_invalid_input(client, field, value):
    payload = {"luas_m2": 100, "jumlah_kamar": 3, "usia_tahun": 5}
    payload[field] = value
    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize("field", ["luas_m2", "jumlah_kamar", "usia_tahun"])
def test_missing_field(client, field):
    payload = {"luas_m2": 100, "jumlah_kamar": 3, "usia_tahun": 5}
    del payload[field]
    assert client.post("/predict", json=payload).status_code == 422


def test_health_and_docs(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_ready": True}
    assert client.get("/docs").status_code == 200
    assert "/predict" in client.get("/openapi.json").json()["paths"]


def test_missing_model(monkeypatch):
    def missing(*args):
        raise FileNotFoundError("Belum training")

    monkeypatch.setattr(main, "load_model", missing)
    with TestClient(main.app) as client:
        responses = [client.get("/health"), client.post("/predict", json={
            "luas_m2": 100, "jumlah_kamar": 3, "usia_tahun": 5,
        })]
        for response in responses:
            assert response.status_code == 503
            assert "python train.py" in response.json()["detail"]
        assert client.get("/docs").status_code == 200


def test_model_loaded_once(monkeypatch, trained):
    calls = []

    def load_once(*args):
        calls.append(1)
        return trained[0]

    monkeypatch.setattr(main, "load_model", load_once)
    with TestClient(main.app) as client:
        for _ in range(2):
            assert client.post("/predict", json={
                "luas_m2": 100, "jumlah_kamar": 3, "usia_tahun": 5,
            }).status_code == 200
    assert len(calls) == 2
