import os
from time import time
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, Body
from pydantic import BaseModel

import mlflow
from mlflow.tracking import MlflowClient
import mlflow.xgboost

from app.config import logging, Config

# ─── OpenTelemetry Metrics for Prometheus ──────────────────
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.prometheus import PrometheusMetricReader, start_http_server

# Init Prometheus OTEL export
reader = PrometheusMetricReader()
provider = MeterProvider(metric_readers=[reader])
metrics.set_meter_provider(provider)
meter = metrics.get_meter_provider().get_meter("prediction_api")
start_http_server(port=8001, addr="0.0.0.0")

# Metrics holder
last_avg_entropy = 0.0
last_avg_confidence = 0.0

# Observable gauges
meter.create_observable_gauge(
    name="api_prediction_entropy",
    callbacks=[lambda options: [metrics.Observation(last_avg_entropy)]],
    description="Average prediction entropy"
)
meter.create_observable_gauge(
    name="api_avg_confidence",
    callbacks=[lambda options: [metrics.Observation(last_avg_confidence)]],
    description="Average model confidence"
)

# ─── ENV ────────────────────────────────────────────────────
os.environ['AWS_ACCESS_KEY_ID'] = os.getenv("AWS_ACCESS_KEY_ID", "minio")
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv("AWS_SECRET_ACCESS_KEY", "minio123")
os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio-service.kubeflow.svc.cluster.local:9000")

# ─── Load Model ─────────────────────────────────────────────
mlflow.set_tracking_uri(Config.MLFLOW_URI)
client = MlflowClient()
versions = client.get_latest_versions("v1_XGB", stages=["Production"]) or client.get_latest_versions("v1_XGB", stages=["None"])
model_uri = f"models:/v1_XGB/{versions[0].version}"
logging.info(f"Loading model: {model_uri}")
xgb_model = mlflow.xgboost.load_model(model_uri)

# ─── FastAPI ────────────────────────────────────────────────
app = FastAPI()

# ─── Feature Schema ─────────────────────────────────────────
FEATURES = [
    "CODE_GENDER", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
    "ORGANIZATION_TYPE", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "REGION_POPULATION_RELATIVE", "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH", "FLAG_EMP_PHONE", "REGION_RATING_CLIENT",
    "REGION_RATING_CLIENT_W_CITY", "REG_CITY_NOT_LIVE_CITY", "REG_CITY_NOT_WORK_CITY",
    "EXT_SOURCE_2", "DAYS_LAST_PHONE_CHANGE", "FLAG_DOCUMENT_3"
]

class DataItem(BaseModel):
    CODE_GENDER: float
    NAME_INCOME_TYPE: float
    NAME_EDUCATION_TYPE: float
    NAME_FAMILY_STATUS: float
    ORGANIZATION_TYPE: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    AMT_GOODS_PRICE: float
    REGION_POPULATION_RELATIVE: float
    DAYS_BIRTH: float
    DAYS_EMPLOYED: float
    DAYS_REGISTRATION: float
    DAYS_ID_PUBLISH: float
    FLAG_EMP_PHONE: float
    REGION_RATING_CLIENT: float
    REGION_RATING_CLIENT_W_CITY: float
    REG_CITY_NOT_LIVE_CITY: float
    REG_CITY_NOT_WORK_CITY: float
    EXT_SOURCE_2: float
    DAYS_LAST_PHONE_CHANGE: float
    FLAG_DOCUMENT_3: float

def compute_entropy(probs: np.ndarray) -> float:
    return float(-np.sum(probs * np.log2(probs + 1e-10)))

def compute_confidence(probs: np.ndarray) -> float:
    return float(np.max(probs))

@app.get("/")
def home() -> Dict[str, str]:
    return {"status": "ok", "message": "API is working."}

@app.post("/Prediction")
async def underwrite_predict(input: List[DataItem] = Body(...)) -> Dict[str, Any]:
    global last_avg_entropy, last_avg_confidence

    start = time()
    df = pd.DataFrame([i.dict() for i in input])
    X = df[FEATURES]

    proba = xgb_model.predict_proba(X)
    preds = np.argmax(proba, axis=1)

    results = []
    entropies, confidences = [], []

    for p, pr in zip(preds, proba):
        entropy = compute_entropy(pr)
        confidence = compute_confidence(pr)

        entropies.append(entropy)
        confidences.append(confidence)

        results.append({
            "result": "Accept" if p == 0 else "Decline",
            "prob_accept": float(pr[0]),
            "prob_decline": float(pr[1]),
            "entropy": round(entropy, 4),
            "confidence": round(confidence, 4)
        })

    # Record metrics for Prometheus
    last_avg_entropy = float(np.mean(entropies))
    last_avg_confidence = float(np.mean(confidences))

    logging.info(f"[Prediction] Entropy={last_avg_entropy:.4f}, Confidence={last_avg_confidence:.3f}")

    return {
        "predictions": results,
        "metrics": {
            "avg_entropy": last_avg_entropy,
            "avg_confidence": last_avg_confidence
        }
    }
