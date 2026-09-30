"""Penyimpanan dan prediksi model yang digunakan oleh training serta API."""

from pathlib import Path

import joblib
from sklearn.linear_model import LinearRegression


# Path tetap benar meskipun program dijalankan dari folder lain.
MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "house_model.joblib"


def save_model(model: LinearRegression, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path = MODEL_PATH) -> LinearRegression:
    # Muat hanya file hasil training sendiri; joblib bukan untuk file tak tepercaya.
    return joblib.load(path)


def predict_price(
    model: LinearRegression, luas_m2: float, jumlah_kamar: int, usia_tahun: float
) -> float:
    # Urutan fitur harus sama dengan urutan kolom ketika training.
    features = [[luas_m2, jumlah_kamar, usia_tahun]]
    return round(float(model.predict(features)[0]), 2)
