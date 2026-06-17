from contextlib import asynccontextmanager
from typing import Annotated

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from data_prep import combine_features
from paths import MODEL_PATH

_model = None


def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    _model = load_model()
    yield


app = FastAPI(
    title="Weather Bias Correction API",
    description="Predict ECMWF 2m-temperature forecast error using time of day and soil temperature.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    time_of_day: Annotated[float, Field(ge=0, le=24, description="Local time in decimal hours")]
    soil_temperature: float = Field(description="Model soil surface temperature (°C)")
    raw_forecast_temp: float | None = Field(
        default=None,
        description="Optional raw 2m forecast temperature (°C) to compute a bias-corrected value",
    )


class PredictResponse(BaseModel):
    predicted_forecast_error: float
    corrected_temperature: float | None = None


class BatchPredictRequest(BaseModel):
    samples: list[PredictRequest] = Field(min_length=1)


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str


def _predict_one(sample: PredictRequest) -> PredictResponse:
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not found at {MODEL_PATH}. Run train_pipeline.py first.",
        )

    features = combine_features(
        np.array([sample.time_of_day]),
        np.array([sample.soil_temperature]),
    )
    predicted_error = float(_model.predict(features)[0])

    corrected = None
    if sample.raw_forecast_temp is not None:
        corrected = sample.raw_forecast_temp - predicted_error

    return PredictResponse(
        predicted_forecast_error=predicted_error,
        corrected_temperature=corrected,
    )


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
        model_path=str(MODEL_PATH),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    return _predict_one(request)


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest):
    return BatchPredictResponse(predictions=[_predict_one(s) for s in request.samples])
