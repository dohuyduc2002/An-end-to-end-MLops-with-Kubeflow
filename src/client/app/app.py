from typing import List, Dict, Any, Optional
from time import time

import numpy as np
import pandas as pd
from fastapi import FastAPI, Body
from pydantic import BaseModel

import mlflow
from mlflow.tracking import MlflowClient

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.prometheus import PrometheusMetricReader, start_http_server

from pathlib import Path
import joblib
from minio import Minio
from io import BytesIO
from loguru import logger
from dotenv import load_dotenv
import os

from schema import RawItem
load_dotenv(override=False)

access_key = os.getenv("MINIO_ACCESS_KEY")
secret_key = os.getenv("MINIO_SECRET_KEY")
minio_endpoint = os.getenv("MINIO_ENDPOINT") 


os.environ["AWS_ACCESS_KEY_ID"] = access_key
os.environ["AWS_SECRET_ACCESS_KEY"] = secret_key
os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://{minio_endpoint}"

minio_client = Minio(
    minio_endpoint,
    access_key=access_key,
    secret_key=secret_key,
    secure=False,
)

local_path = Path(__file__).resolve().parents[1] / "joblib" / "transformer.joblib"
docker_path = Path("/app/joblib/transformer.joblib")

if local_path.exists():
    transformer = joblib.load(local_path)
elif docker_path.exists():
    transformer = joblib.load(docker_path)
else:
    raise FileNotFoundError("transformer.joblib not found.")


model_name = os.getenv("MODEL_NAME")
model_type = os.getenv("MODEL_TYPE")

mlflow_uri = os.getenv("MLFLOW_ENDPOINT")
mlflow.set_tracking_uri(mlflow_uri)

client = MlflowClient()
versions = (
    client.get_latest_versions(model_name, stages=["Production"])
    or client.get_latest_versions(model_name, stages=["None"])
)

model_uri = f"models:/{model_name}/{versions[0].version}"

if model_type == "xgb":
    model = mlflow.xgboost.load_model(model_uri)
elif model_type == "lgbm":
    model = mlflow.lightgbm.load_model(model_uri)
else:
    raise ValueError(f"Unsupported model type: {model_type}")

logger.info(f"Loaded {model_type.upper()} model '{model_name}' from {model_uri}")

# ========== OpenTelemetry gauges ===========================
reader = PrometheusMetricReader()
provider = MeterProvider(metric_readers=[reader])
metrics.set_meter_provider(provider)
meter = metrics.get_meter_provider().get_meter("prediction_api")
start_http_server(port=8001, addr="0.0.0.0")

last_avg_entropy = 0.0
last_avg_confidence = 0.0

meter.create_observable_gauge(
    name="api_prediction_entropy",
    callbacks=[lambda opts: [metrics.Observation(last_avg_entropy)]],
    description="Average prediction entropy",
)
meter.create_observable_gauge(
    name="api_avg_confidence",
    callbacks=[lambda opts: [metrics.Observation(last_avg_confidence)]],
    description="Average model confidence",
)

# ========== FastAPI ========================================
app = FastAPI()


def entropy(p: np.ndarray) -> float:
    return float(-np.sum(p * np.log2(p + 1e-10)))

def confidence(p: np.ndarray) -> float:
    return float(p.max())

@app.get("/")
def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.post("/Prediction")
async def predict(items: List[RawItem] = Body(...)) -> Dict[str, Any]:
    global last_avg_entropy, last_avg_confidence
    t0 = time()
    df_raw = pd.DataFrame([i.dict() for i in items]).replace({None: np.nan})

    binning = transformer["binning_process"]
    selector = transformer["selector"]

    df_raw = df_raw[[col for col in df_raw.columns if col in binning.variable_names]]

    # pipeline
    X_binned = binning.transform(df_raw)
    X = selector.transform(X_binned)

    proba = model.predict_proba(X)

    preds = np.argmax(proba, axis=1)

    entropies = [entropy(p) for p in proba]
    confidences = [confidence(p) for p in proba]

    last_avg_entropy = float(np.mean(entropies))
    last_avg_confidence = float(np.mean(confidences))

    return {
        "inference_time_ms": round((time() - t0) * 1000, 2),
        "predictions": [
            {
                "result": "Accept" if y == 0 else "Decline",
                "prob_accept": float(p[0]),
                "prob_decline": float(p[1]),
                "entropy": round(e, 4),
                "confidence": round(c, 4),
            }
            for y, p, e, c in zip(preds, proba, entropies, confidences)
        ],
        "metrics": {
            "avg_entropy": last_avg_entropy,
            "avg_confidence": last_avg_confidence,
        },
    }

@app.post("/Prediction-by-id")
def predict_by_id(id: int) -> Dict[str, Any]:
    global last_avg_entropy, last_avg_confidence
    t0 = time()

    try:
        response = minio_client.get_object("sample-data", "data/application_test.csv")
        df_all = pd.read_csv(BytesIO(response.read()))
    except Exception as e:
        raise RuntimeError(f"Failed to fetch test data from MinIO: {str(e)}")

    df_row = df_all[df_all["SK_ID_CURR"] == id]
    if df_row.empty:
        return {"error": f"ID {id} not found"}

    binning = transformer["binning_process"]
    selector = transformer["selector"]

    df_row = df_row[[col for col in df_row.columns if col in binning.variable_names]]

    X_binned = binning.transform(df_row)
    X = selector.transform(X_binned)

    proba = model.predict_proba(X)
    preds = np.argmax(proba, axis=1)

    entropies = [entropy(p) for p in proba]
    confidences = [confidence(p) for p in proba]

    last_avg_entropy = float(np.mean(entropies))
    last_avg_confidence = float(np.mean(confidences))

    return {
        "inference_time_ms": round((time() - t0) * 1000, 2),
        "predictions": [
            {
                "result": "Accept" if y == 0 else "Decline",
                "prob_accept": float(p[0]),
                "prob_decline": float(p[1]),
                "entropy": round(e, 4),
                "confidence": round(c, 4),
            }
            for y, p, e, c in zip(preds, proba, entropies, confidences)
        ],
        "metrics": {
            "avg_entropy": last_avg_entropy,
            "avg_confidence": last_avg_confidence,
        },
    }
