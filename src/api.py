"""Small, validated HTTP API for the trained fraud-classification pipeline."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any
import math

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(
        ..., description="Feature-name to numeric-value mapping from one transaction."
    )


class PredictionResponse(BaseModel):
    prediction: int
    fraud_probability: float
    threshold: float
    model_version: str


def create_app(model_path: str | Path | None = None) -> FastAPI:
    resolved_model_path = Path(
        model_path or os.getenv("MODEL_PATH", "artifacts/fraud_pipeline.joblib")
    )
    application = FastAPI(
        title="Fraud Detection API",
        version="1.0.0",
        description="Local portfolio API for a fraud-classification model.",
    )
    application.state.model = None
    application.state.model_path = resolved_model_path
    application.state.feature_names = None

    def get_model() -> Any:
        if application.state.model is not None:
            return application.state.model
        path = application.state.model_path
        if not path.is_file():
            raise HTTPException(
                status_code=503,
                detail=f"Model artifact not found at {path}. Train the model first.",
            )
        try:
            application.state.model = joblib.load(path)
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Could not load model artifact.") from exc
        return application.state.model

    @application.get("/health")
    def health() -> dict[str, str | bool]:
        path = application.state.model_path
        ready = application.state.model is not None or path.is_file()
        return {"status": "ok" if ready else "not_ready", "model_available": ready}

    @application.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> PredictionResponse:
        model = get_model()
        expected = getattr(model, "feature_names_in_", None)
        if expected is None:
            expected = application.state.feature_names
        if expected is None:
            raise HTTPException(
                status_code=500,
                detail="Model does not expose feature names; cannot validate request schema.",
            )
        expected_names = [str(name) for name in expected]
        supplied_names = set(request.features)
        expected_set = set(expected_names)
        missing = sorted(expected_set - supplied_names)
        extra = sorted(supplied_names - expected_set)
        if missing or extra:
            raise HTTPException(
                status_code=422,
                detail={"missing_features": missing, "unexpected_features": extra},
            )
        values = [request.features[name] for name in expected_names]
        if not all(math.isfinite(value) for value in values):
            raise HTTPException(status_code=422, detail="Feature values must be finite numbers.")

        probability = float(model.predict_proba(
            __import__("pandas").DataFrame([values], columns=expected_names)
        )[0][1])
        threshold = float(os.getenv("FRAUD_THRESHOLD", "0.5"))
        path = application.state.model_path
        version = (
            hashlib.sha256(path.read_bytes()).hexdigest()[:12]
            if path.is_file()
            else "in-memory"
        )
        return PredictionResponse(
            prediction=int(probability >= threshold),
            fraud_probability=probability,
            threshold=threshold,
            model_version=version,
        )

    return application


app = create_app()
