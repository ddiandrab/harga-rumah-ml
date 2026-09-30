"""Buat data sintetis, latih regresi linear, evaluasi, lalu simpan model."""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from model import MODEL_PATH, save_model


def generate_data(n_samples: int = 500, seed: int = 42):
    rng = np.random.default_rng(seed)
    luas = rng.uniform(20, 500, n_samples)
    kamar = rng.integers(1, 11, n_samples)
    usia = rng.uniform(0, 50, n_samples)
    noise = rng.normal(0, 25_000_000, n_samples)

    # Rumus ini sengaja dibuat untuk belajar, bukan berasal dari pasar properti.
    harga = 200_000_000 + luas * 8_000_000 + kamar * 40_000_000 - usia * 2_000_000 + noise
    features = np.column_stack([luas, kamar, usia])
    return features, harga


def train_model():
    features, target = generate_data()
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    # Pembanding sederhana: selalu menebak rata-rata harga dari data training.
    baseline = np.full(y_test.shape, y_train.mean())
    metrics = {
        "mae_rupiah": float(mean_absolute_error(y_test, predictions)),
        "baseline_mae_rupiah": float(mean_absolute_error(y_test, baseline)),
        "train_samples": len(x_train),
        "test_samples": len(x_test),
    }
    return model, metrics


def main():
    model, metrics = train_model()
    save_model(model)
    print(f"Data training: {metrics['train_samples']}; data pengujian: {metrics['test_samples']}")
    print(f"MAE model: Rp {metrics['mae_rupiah']:,.2f}")
    print(f"MAE pembanding rata-rata: Rp {metrics['baseline_mae_rupiah']:,.2f}")
    print(f"Model disimpan di: {MODEL_PATH}")
    print("Data sintetis: hasil ini bukan ukuran akurasi harga rumah nyata.")


if __name__ == "__main__":
    main()
