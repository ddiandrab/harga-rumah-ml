"""Fixture buatan hanya untuk pengujian, bukan dataset rumah nyata."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import main
from data_processing import FEATURES, read_houses
from model import load_model, predict_house, save_model
from train import synthetic_bundle, train_jabodetabek


@pytest.fixture
def csv_file(tmp_path):
    rng = np.random.default_rng(8)
    rows = []
    for i in range(100):
        land = float(rng.integers(40, 300))
        building = float(rng.integers(30, 200))
        rows.append({"ads_id": f"id-{i}", "city": "  bekasi " if i % 2 else "Bogor",
                     "land_size_m2": land, "building_size_m2": building,
                     "bedrooms": int(rng.integers(1, 6)), "bathrooms": int(rng.integers(1, 4)),
                     "price_in_rp": land * 4e6 + building * 3e6 + (i % 2) * 1e8})
    path = tmp_path / "houses.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def bundle(csv_file):
    return train_jabodetabek(csv_file)


@pytest.fixture
def client(monkeypatch, bundle):
    monkeypatch.setattr(main, "load_model", lambda *args: bundle if args else synthetic_bundle())
    with TestClient(main.app) as client:
        yield client


def test_cleaning(csv_file):
    frame = pd.read_csv(csv_file)
    extra = frame.iloc[:5].copy().astype({"price_in_rp": "object", "bedrooms": "float"})
    extra.loc[0, "city"] = " "
    extra.loc[1, "price_in_rp"] = "invalid"
    extra.loc[2, "bedrooms"] = 1.5
    extra.loc[3, "land_size_m2"] = np.inf
    # Fifth row has the same ads_id, even though its price differs.
    extra.loc[4, "price_in_rp"] = 1e9
    pd.concat([frame, extra], ignore_index=True).to_csv(csv_file, index=False)
    clean, report = read_houses(csv_file)
    assert report == {"raw_rows": 105, "missing_or_unreadable_rows": 2,
                      "invalid_rows": 2, "duplicate_rows": 1, "clean_rows": 100}
    assert set(clean.city) == {"Bekasi", "Bogor"}


def test_missing_columns_and_small_data(csv_file):
    pd.DataFrame({"city": ["Bekasi"]}).to_csv(csv_file, index=False)
    with pytest.raises(ValueError, match="Kolom CSV wajib"):
        read_houses(csv_file)
    pd.DataFrame(columns=FEATURES + ["price_in_rp"]).to_csv(csv_file, index=False)
    with pytest.raises(ValueError, match="minimal 30"):
        read_houses(csv_file)


def test_duplicate_without_id(csv_file):
    data = pd.read_csv(csv_file).drop(columns="ads_id")
    pd.concat([data, data.iloc[:1]], ignore_index=True).to_csv(csv_file, index=False)
    assert read_houses(csv_file)[1]["duplicate_rows"] == 1


def test_bundle_training_roundtrip(bundle, tmp_path):
    meta = bundle["metadata"]
    assert meta["split"] == {"train": 60, "validation": 20, "test": 20, "final_fit": 80}
    assert meta["cities"] == ["Bekasi", "Bogor"]
    assert set(meta["validation_metrics"]) == {"LinearRegression", "RandomForest", "MedianBaseline"}
    selected = min(["LinearRegression", "RandomForest"], key=lambda name: meta["validation_metrics"][name]["mae_rupiah"])
    assert meta["model_name"] == selected
    assert meta["test_metrics"]["mae_rupiah"] < meta["baseline_test_metrics"]["mae_rupiah"]
    path = tmp_path / "bundle.joblib"
    save_model(bundle, path)
    assert predict_house(load_model(path), meta["example"]) == predict_house(bundle, meta["example"])
    assert meta["versions"]["python"].startswith("3.12")


PAYLOAD = {"kota": " bekasi ", "luas_tanah_m2": 100, "luas_bangunan_m2": 80,
           "kamar_tidur": 3, "kamar_mandi": 2}


def test_prediction_and_models(client):
    response = client.post("/predict/jabodetabek", json=PAYLOAD)
    assert response.status_code == 200
    assert response.json()["prediksi_harga_rupiah"] > 0
    assert response.json()["warnings"] == []
    metadata = client.get("/models").json()
    assert metadata["jabodetabek"]["ready"]
    assert metadata["jabodetabek"]["metadata"]["cities"] == ["Bekasi", "Bogor"]
    assert "estimator" not in metadata["jabodetabek"]
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200


@pytest.mark.parametrize("key,value", [
    ("kota", "Bandung"), ("kota", " "), ("kota", 123),
    ("luas_tanah_m2", 0), ("luas_bangunan_m2", -1), ("luas_tanah_m2", "100"),
    ("kamar_tidur", 1.5), ("kamar_mandi", True), ("kamar_mandi", 0),
])
def test_validation(client, key, value):
    assert client.post("/predict/jabodetabek", json={**PAYLOAD, key: value}).status_code == 422


def test_missing_and_extra_field(client):
    for key in PAYLOAD:
        payload = PAYLOAD.copy()
        del payload[key]
        assert client.post("/predict/jabodetabek", json=payload).status_code == 422
    assert client.post("/predict/jabodetabek", json={**PAYLOAD, "extra": 1}).status_code == 422


def test_nonfinite(client):
    raw = '{"kota":"Bekasi","luas_tanah_m2":1e309,"luas_bangunan_m2":80,"kamar_tidur":3,"kamar_mandi":2}'
    response = client.post("/predict/jabodetabek", content=raw, headers={"Content-Type": "application/json"})
    assert response.status_code == 422


def test_range_warning(client):
    response = client.post("/predict/jabodetabek", json={**PAYLOAD, "luas_tanah_m2": 500})
    assert response.status_code == 200
    assert "Luas tanah" in response.json()["warnings"][0]


def test_unavailable_and_corrupt_models(monkeypatch):
    for exception in [FileNotFoundError, ValueError]:
        def fail(*args):
            raise exception("Model tidak tersedia")
        monkeypatch.setattr(main, "load_model", fail)
        with TestClient(main.app) as client:
            assert client.get("/").status_code == 200
            assert client.get("/docs").status_code == 200
            assert not client.get("/models").json()["jabodetabek"]["ready"]
            response = client.post("/predict/jabodetabek", json=PAYLOAD)
            assert response.status_code == 503
            assert "--dataset jabodetabek" in response.json()["detail"]
