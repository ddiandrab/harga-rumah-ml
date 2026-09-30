"""REST API lokal: jalankan dengan python -m uvicorn main:app --reload."""

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from model import load_model, predict_price


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Model dimuat sekali, bukan setiap kali menerima permintaan.
    try:
        app.state.model = load_model()
    except FileNotFoundError:
        app.state.model = None
        logging.warning("Model belum tersedia. Jalankan python train.py lalu restart API.")
    yield


app = FastAPI(
    title="Belajar ML: Prediksi Harga Rumah",
    description="Prediksi dari data sintetis untuk latihan, bukan harga pasar nyata.",
    lifespan=lifespan,
)


class HouseInput(BaseModel):
    """Skema JSON sekaligus aturan validasi masukan pengguna."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {"luas_m2": 100, "jumlah_kamar": 3, "usia_tahun": 5}
        },
    )
    luas_m2: float = Field(ge=20, le=500, strict=True, allow_inf_nan=False)
    jumlah_kamar: int = Field(ge=1, le=10, strict=True)
    usia_tahun: float = Field(ge=0, le=50, strict=True, allow_inf_nan=False)


class PredictionOutput(BaseModel):
    prediksi_harga_rupiah: float


def require_model():
    if app.state.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model belum tersedia. Jalankan python train.py lalu restart API.",
        )
    return app.state.model


@app.get("/health")
def health():
    require_model()
    return {"status": "ok", "model_ready": True}


@app.post("/predict", response_model=PredictionOutput)
def predict(house: HouseInput):
    trained_model = require_model()
    price = predict_price(
        trained_model, house.luas_m2, house.jumlah_kamar, house.usia_tahun
    )
    return PredictionOutput(prediksi_harga_rupiah=price)
