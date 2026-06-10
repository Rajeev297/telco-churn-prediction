import os
import pickle
import time
import pandas as pd
import numpy as np
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import FileResponse
from typing import List, Dict

from api.schemas import (
    CustomerData,
    PredictionRequest,
    SinglePrediction,
    PredictionResponse,
    HealthResponse,
)
from api.audit import log_api_event, log_prediction
from src.config import get


router = APIRouter()

MODELS_DIR = get("paths.models_dir", "models")
REPORTS_DIR = get("paths.reports_dir", "reports")

_model = None
_preprocessor = None
_threshold = 0.5
_model_name = "unknown"


def load_api_artifacts():
    global _model, _preprocessor, _threshold, _model_name
    if _model is not None:
        return

    prep_path = os.path.join(MODELS_DIR, "preprocessor.pkl")
    if os.path.exists(prep_path):
        with open(prep_path, "rb") as f:
            _preprocessor = pickle.load(f)

    files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pkl") and "preprocessor" not in f and "explainer" not in f and "metadata" not in f]
    if files:
        latest = max(files, key=lambda f: os.path.getctime(os.path.join(MODELS_DIR, f)))
        with open(os.path.join(MODELS_DIR, latest), "rb") as f:
            obj = pickle.load(f)
            if isinstance(obj, dict) and "model" in obj:
                _model = obj["model"]
                _threshold = obj.get("threshold", 0.5)
                _model_name = obj.get("model_name", "unknown")
            else:
                _model = obj
                _model_name = "unknown"

    if _model is None:
        raise RuntimeError("No trained model found")


def _get_risk(prob: float) -> str:
    if prob < 0.3:
        return "Low"
    if prob < 0.7:
        return "Medium"
    return "High"


@router.get("/health", response_model=HealthResponse)
async def health():
    try:
        load_api_artifacts()
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            model_name=_model_name,
            threshold=float(_threshold),
        )
    except Exception as e:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            model_name="",
            threshold=0.5,
        )


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, body: PredictionRequest):
    start = time.time()
    client_ip = request.client.host if request.client else "unknown"

    try:
        load_api_artifacts()
    except RuntimeError as e:
        log_api_event("predict", client_ip, "error", {"error": str(e)})
        raise HTTPException(status_code=503, detail=str(e))

    try:
        records = [c.model_dump() for c in body.customers]
        df = pd.DataFrame(records)
        X_processed = _preprocessor.transform(df)
        probs = _model.predict_proba(X_processed)[:, 1]
        preds = (probs >= _threshold).astype(int)

        predictions: List[SinglePrediction] = []
        high_risk = 0
        total_prob = 0.0
        for prob, pred in zip(probs, preds):
            risk = _get_risk(prob)
            predictions.append(SinglePrediction(
                churn_probability=round(float(prob), 4),
                churn_prediction="Yes" if pred else "No",
                risk_level=risk,
            ))
            if risk == "High":
                high_risk += 1
            total_prob += prob

        n = len(predictions)
        summary = {
            "total_customers": n,
            "predicted_churn": int(preds.sum()),
            "churn_rate": round(float(preds.mean() * 100), 2),
            "avg_churn_probability": round(float(total_prob / n), 4) if n else 0,
            "high_risk_count": high_risk,
            "threshold": float(_threshold),
        }

        elapsed = (time.time() - start) * 1000
        log_prediction(client_ip, n, high_risk, total_prob / n, elapsed)

        return PredictionResponse(predictions=predictions, summary=summary)

    except Exception as e:
        log_api_event("predict", client_ip, "error", {"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
