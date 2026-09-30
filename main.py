"""API dan halaman belajar, dijalankan dengan python -m uvicorn main:app --reload."""
from contextlib import asynccontextmanager
import logging
import math

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from data_processing import normalize_city
from model import ROOT, JABODETABEK_PATH, load_model, predict_price, predict_house

COMMANDS = {
    "synthetic": "python train.py",
    "jabodetabek": "python train.py --dataset jabodetabek --csv data/raw/jabodetabek_house_price.csv",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = {}
    app.state.errors = {}
    for name in COMMANDS:
        try:
            bundle = load_model() if name == "synthetic" else load_model(JABODETABEK_PATH)
            if name == "jabodetabek" and (
                not isinstance(bundle, dict) or "estimator" not in bundle or "metadata" not in bundle
            ):
                raise ValueError("Format artefak tidak cocok")
            app.state.models[name] = bundle
        except Exception:
            # Kegagalan satu artefak tidak mematikan halaman atau mode lain.
            logging.exception("Model %s belum tersedia atau gagal dimuat", name)
            app.state.models[name] = None
            app.state.errors[name] = f"Model belum tersedia atau gagal dimuat. Jalankan {COMMANDS[name]} lalu restart API."
    yield


app = FastAPI(title="RumahLab · Belajar Python & ML", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.exception_handler(RequestValidationError)
async def validation_error(request, error):
    # Jangan salin masukan Infinity/NaN ke JSON respons: JSON tidak mendukungnya.
    details = [{key: item[key] for key in ("loc", "msg", "type")}
               for item in error.errors()]
    return JSONResponse(status_code=422, content={"detail": details})


class HouseInput(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={
        "example": {"luas_m2": 100, "jumlah_kamar": 3, "usia_tahun": 5}})
    luas_m2: float = Field(ge=20, le=500, strict=True, allow_inf_nan=False)
    jumlah_kamar: int = Field(ge=1, le=10, strict=True)
    usia_tahun: float = Field(ge=0, le=50, strict=True, allow_inf_nan=False)


class JabodetabekInput(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "kota": "Bekasi", "luas_tanah_m2": 100, "luas_bangunan_m2": 80,
        "kamar_tidur": 3, "kamar_mandi": 2}})
    kota: str = Field(min_length=1, strict=True)
    luas_tanah_m2: float = Field(gt=0, strict=True, allow_inf_nan=False)
    luas_bangunan_m2: float = Field(gt=0, strict=True, allow_inf_nan=False)
    kamar_tidur: int = Field(gt=0, strict=True)
    kamar_mandi: int = Field(gt=0, strict=True)


class PredictionOutput(BaseModel):
    prediksi_harga_rupiah: float


class JabodetabekOutput(PredictionOutput):
    model: str
    warnings: list[str]


def require_model(name="synthetic"):
    bundle = app.state.models[name]
    if bundle is None:
        raise HTTPException(503, app.state.errors[name])
    return bundle


@app.get("/", include_in_schema=False)
def homepage():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/health")
def health():
    # Pertahankan kontrak health latihan pertama.
    require_model()
    return {"status": "ok", "model_ready": True}


@app.get("/models")
def models():
    result = {}
    for name, bundle in app.state.models.items():
        result[name] = {"ready": bundle is not None, "training_command": COMMANDS[name]}
        if bundle is None:
            result[name]["message"] = app.state.errors[name]
        elif isinstance(bundle, dict):
            result[name]["metadata"] = bundle["metadata"]
        else:
            result[name]["metadata"] = {"model_name": "LinearRegression", "cities": [],
                "message": "Artefak lama. Latih ulang untuk melengkapi metadata evaluasi."}
    return result


@app.post("/predict", response_model=PredictionOutput)
def predict(house: HouseInput):
    price = predict_price(require_model(), house.luas_m2, house.jumlah_kamar, house.usia_tahun)
    return PredictionOutput(prediksi_harga_rupiah=price)


@app.post("/predict/jabodetabek", response_model=JabodetabekOutput)
def predict_jabodetabek(house: JabodetabekInput):
    bundle = require_model("jabodetabek")
    metadata = bundle["metadata"]
    city = normalize_city(house.kota)
    if city not in metadata["cities"]:
        raise HTTPException(422, "Kota belum didukung. Lihat daftar kota di GET /models.")
    values = {"city": city, "land_size_m2": house.luas_tanah_m2,
              "building_size_m2": house.luas_bangunan_m2, "bedrooms": house.kamar_tidur,
              "bathrooms": house.kamar_mandi}
    labels = {"land_size_m2": "Luas tanah", "building_size_m2": "Luas bangunan",
              "bedrooms": "Kamar tidur", "bathrooms": "Kamar mandi"}
    warnings = []
    for feature, (low, high) in metadata["feature_ranges"].items():
        if not low <= values[feature] <= high:
            warnings.append(f"{labels[feature]} di luar rentang training ({low:g}–{high:g}); prediksi kurang andal.")
    try:
        price = predict_house(bundle, values)
    except (ValueError, OverflowError):
        raise HTTPException(422, "Angka masukan terlalu ekstrem untuk model. Gunakan rentang training.")
    if not math.isfinite(price) or price <= 0:
        raise HTTPException(422, "Model tidak menghasilkan harga positif yang valid untuk kombinasi ini. Coba masukan dalam rentang training.")
    return JabodetabekOutput(prediksi_harga_rupiah=price, model=metadata["model_name"], warnings=warnings)
