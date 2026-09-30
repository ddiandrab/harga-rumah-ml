"""Membaca dan membersihkan CSV; tidak ada unduhan saat API berjalan."""
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["city", "land_size_m2", "building_size_m2", "bedrooms", "bathrooms"]
NUMERIC_FEATURES = FEATURES[1:]
TARGET = "price_in_rp"
SOURCE = "https://www.kaggle.com/datasets/nafisbarizki/daftar-harga-rumah-jabodetabek"


def normalize_city(value: str) -> str:
    return " ".join(value.strip().split()).title()


def read_houses(path: Path):
    data = pd.read_csv(path)
    missing = set(FEATURES + [TARGET]) - set(data.columns)
    if missing:
        raise ValueError(f"Kolom CSV wajib belum ada: {', '.join(sorted(missing))}")
    report = {"raw_rows": len(data)}
    data["city"] = data["city"].astype("string").str.strip().replace("", pd.NA)
    for column in NUMERIC_FEATURES + [TARGET]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    before = len(data)
    data = data.dropna(subset=FEATURES + [TARGET]).copy()
    report["missing_or_unreadable_rows"] = before - len(data)
    valid = np.isfinite(data[NUMERIC_FEATURES + [TARGET]]).all(axis=1)
    valid &= (data[NUMERIC_FEATURES + [TARGET]] > 0).all(axis=1)
    valid &= (data[["bedrooms", "bathrooms"]] % 1 == 0).all(axis=1)
    before = len(data)
    data = data.loc[valid].copy()
    report["invalid_rows"] = before - len(data)
    data["city"] = data["city"].map(normalize_city)
    before = len(data)
    if "ads_id" in data:
        ids = data["ads_id"].astype("string").str.strip()
        duplicated_id = ids.notna() & ids.ne("") & ids.duplicated()
        data = data.loc[~duplicated_id]
    data = data.drop_duplicates(subset=FEATURES + [TARGET])
    report["duplicate_rows"] = before - len(data)
    report["clean_rows"] = len(data)
    if len(data) < 30:
        raise ValueError(f"Hanya {len(data)} baris valid. Sediakan minimal 30 baris unik untuk pembagian data.")
    return data[FEATURES + [TARGET]].reset_index(drop=True), report
