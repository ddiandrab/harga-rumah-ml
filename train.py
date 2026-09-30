"""Latihan dasar: python train.py. Data nyata: tambahkan --dataset jabodetabek --csv PATH."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_processing import FEATURES, NUMERIC_FEATURES, SOURCE, TARGET, read_houses
from model import MODEL_PATH, JABODETABEK_PATH, save_model


def generate_data(n_samples: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    luas = rng.uniform(20, 500, n_samples)
    kamar = rng.integers(1, 11, n_samples)
    usia = rng.uniform(0, 50, n_samples)
    noise = rng.normal(0, 25_000_000, n_samples)
    harga = 200_000_000 + luas * 8_000_000 + kamar * 40_000_000 - usia * 2_000_000 + noise
    return np.column_stack([luas, kamar, usia]), harga


def evaluate(target, predictions):
    return {"mae_rupiah": float(mean_absolute_error(target, predictions)),
            "r2": float(r2_score(target, predictions))}


def training_info():
    return {"trained_at": datetime.now(timezone.utc).isoformat(), "versions": {
        "python": platform.python_version(), "scikit_learn": sklearn.__version__,
        "numpy": np.__version__, "pandas": pd.__version__}}


def train_model():
    features, target = generate_data()
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42)
    model = LinearRegression().fit(x_train, y_train)
    metrics = evaluate(y_test, model.predict(x_test))
    metrics.update({"baseline_mae_rupiah": float(mean_absolute_error(
        y_test, np.full(y_test.shape, y_train.mean()))),
        "train_samples": len(x_train), "test_samples": len(x_test)})
    return model, metrics


def synthetic_bundle():
    estimator, metrics = train_model()
    return {"estimator": estimator, "metadata": {
        **training_info(), "dataset": "synthetic", "model_name": "LinearRegression",
        "source": "Data sintetis buatan untuk belajar", "feature_order": [
            "luas_m2", "jumlah_kamar", "usia_tahun"], "cities": [],
        "feature_ranges": {"luas_m2": [20, 500], "jumlah_kamar": [1, 10], "usia_tahun": [0, 50]},
        "test_metrics": metrics, "rows": 1000}}


def make_pipeline(estimator):
    preprocessing = ColumnTransformer([
        ("city", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["city"]),
        ("numeric", StandardScaler(), NUMERIC_FEATURES),
    ])
    return Pipeline([("preprocessing", preprocessing), ("model", estimator)])


def train_jabodetabek(csv_path: Path):
    data, cleaning = read_houses(csv_path)
    x, y = data[FEATURES], data[TARGET]
    # Test dipisahkan lebih dahulu dan tidak ikut memilih algoritma.
    x_development, x_test, y_development, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42)
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_development, y_development, test_size=0.25, random_state=42)
    candidates = {
        "LinearRegression": make_pipeline(LinearRegression()),
        "RandomForest": make_pipeline(RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)),
    }
    validation = {}
    for name, pipeline in candidates.items():
        pipeline.fit(x_train, y_train)
        validation[name] = evaluate(y_validation, pipeline.predict(x_validation))
    baseline = DummyRegressor(strategy="median").fit(x_train, y_train)
    validation["MedianBaseline"] = evaluate(y_validation, baseline.predict(x_validation))
    selected = min(candidates, key=lambda name: validation[name]["mae_rupiah"])
    final_model = clone(candidates[selected]).fit(x_development, y_development)
    baseline.fit(x_development, y_development)
    test_metrics = evaluate(y_test, final_model.predict(x_test))
    baseline_metrics = evaluate(y_test, baseline.predict(x_test))
    metadata = {
        **training_info(), "dataset": "jabodetabek", "source": SOURCE,
        "model_name": selected, "feature_order": FEATURES,
        "cities": sorted(x_development["city"].unique().tolist()),
        "feature_ranges": {column: [float(x_development[column].min()),
                                    float(x_development[column].max())] for column in NUMERIC_FEATURES},
        "cleaning": cleaning, "split": {"train": len(x_train), "validation": len(x_validation),
                                        "test": len(x_test), "final_fit": len(x_development)},
        "validation_metrics": validation, "test_metrics": test_metrics,
        "baseline_test_metrics": baseline_metrics,
        "beats_baseline": test_metrics["mae_rupiah"] < baseline_metrics["mae_rupiah"],
        "example": {key: (str(x_development.iloc[0][key]) if key == "city" else
                          float(x_development.iloc[0][key])) for key in FEATURES},
    }
    return {"estimator": final_model, "metadata": metadata}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["synthetic", "jabodetabek"], default="synthetic")
    parser.add_argument("--csv", type=Path, help="Lokasi CSV asli dari Kaggle")
    args = parser.parse_args()
    if args.dataset == "jabodetabek" and args.csv is None:
        parser.error("Mode Jabodetabek memerlukan --csv data/raw/jabodetabek_house_price.csv")
    if args.dataset == "synthetic" and args.csv is not None:
        parser.error("Gunakan --dataset jabodetabek ketika memberikan --csv")
    try:
        bundle = train_jabodetabek(args.csv) if args.dataset == "jabodetabek" else synthetic_bundle()
    except (OSError, ValueError) as error:
        parser.exit(1, f"Training gagal: {error}\nPeriksa file CSV; lihat README untuk langkah persiapan.\n")
    destination = JABODETABEK_PATH if args.dataset == "jabodetabek" else MODEL_PATH
    save_model(bundle, destination)
    print(json.dumps(bundle["metadata"], indent=2, ensure_ascii=False))
    print(f"Model disimpan di {destination}. Restart API untuk memuatnya.")


if __name__ == "__main__":
    main()
