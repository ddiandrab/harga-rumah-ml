"""Artefak lokal menyimpan estimator dan metadata dalam satu file."""
from pathlib import Path
import os
import tempfile

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "house_model.joblib"
JABODETABEK_PATH = ROOT / "artifacts" / "jabodetabek.joblib"


def save_model(model, path: Path = MODEL_PATH) -> None:
    # Ganti file setelah penulisan selesai agar artefak lama tidak terpotong.
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        joblib.dump(model, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_model(path: Path = MODEL_PATH):
    # Joblib hanya untuk artefak tepercaya yang kita latih sendiri.
    return joblib.load(path)


def predict_price(model, luas_m2: float, jumlah_kamar: int, usia_tahun: float) -> float:
    # Artefak latihan versi awal tetap dapat digunakan.
    estimator = model["estimator"] if isinstance(model, dict) else model
    return round(float(estimator.predict([[luas_m2, jumlah_kamar, usia_tahun]])[0]), 2)


def predict_house(bundle, values: dict) -> float:
    frame = pd.DataFrame([values], columns=bundle["metadata"]["feature_order"])
    return round(float(bundle["estimator"].predict(frame)[0]), 2)
