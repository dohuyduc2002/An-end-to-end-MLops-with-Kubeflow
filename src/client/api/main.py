from contextlib import asynccontextmanager
from functools import wraps
from io import BytesIO
from time import time
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, Body, Depends
from opentelemetry import metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from prometheus_client import start_http_server

from schema import RawItem
from utils import ApiConfig, Predictor


def entropy(p: np.ndarray) -> float:
    return float(np.sum(p * np.log2(p + 1e-10)))


def confidence(p: np.ndarray) -> float:
    return float(p.max())


class MetricsHandler:
    def __init__(self) -> None:
        self._avg_entropy = 0.0
        self._avg_confidence = 0.0

        reader = PrometheusMetricReader()
        metrics.set_meter_provider(MeterProvider(metric_readers=[reader]))
        meter = metrics.get_meter_provider().get_meter("prediction_api")
        start_http_server(addr="0.0.0.0", port=8001)

        meter.create_observable_gauge(
            "api_prediction_entropy",
            callbacks=[lambda _opts: [metrics.Observation(self._avg_entropy)]],
        )
        meter.create_observable_gauge(
            "api_avg_confidence",
            callbacks=[lambda _opts: [metrics.Observation(self._avg_confidence)]],
        )

    def update(self, ents: List[float], confs: List[float]) -> None:
        self._avg_entropy = float(np.mean(ents))
        self._avg_confidence = float(np.mean(confs))


# ----------------------------------------------------------------------
# 2) Decorator tính metrics
# ----------------------------------------------------------------------
def otel_metric(fn):
    @wraps(fn)
    def wrapper(self, df: pd.DataFrame):
        result = fn(self, df)
        print("🧪 Result type:", type(result), "| Result:", result)

        if not isinstance(result, tuple) or len(result) != 2:
            print("⚠️ Skipping metrics: result not a (preds, proba) tuple")
            return result

        start_time = time()
        preds, proba = result
        ents = [entropy(p) for p in proba]
        confs = [confidence(p) for p in proba]
        self.metrics.update(ents, confs)
        return self._build_response(start_time, preds, proba, ents, confs)

    return wrapper


# ----------------------------------------------------------------------
# 3) Service lớp chính
# ----------------------------------------------------------------------
class PredictionService:
    def __init__(self, cfg: ApiConfig, predictor: Predictor):
        self.cfg = cfg
        self.predictor = predictor
        self.metrics = MetricsHandler()

    @classmethod
    async def create(cls, cfg: ApiConfig):
        predictor = Predictor(cfg)  
        await predictor.load_artifacts_async()  
        return cls(cfg, predictor)

    # --------------------------------------------------------------
    # helper build response
    # --------------------------------------------------------------
    def _build_response(
        self,
        start_time: float,
        preds: np.ndarray,
        proba: np.ndarray,
        entropies: List[float],
        confidences: List[float],
    ) -> Dict[str, Any]:
        return {
            "inference_time_ms": round((time() - start_time) * 1000, 2),
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
                "avg_entropy": self.metrics._avg_entropy,
                "avg_confidence": self.metrics._avg_confidence,
            },
        }

    @otel_metric
    def predict_items(self, items: List[RawItem]):
        df = pd.DataFrame([i.dict() for i in items]).replace({None: np.nan})
        return self.predictor.inference(df)

    @otel_metric
    def predict_by_id(self, sk_id: int):
        minio_client = self.cfg.get_minio_client()
        response = minio_client.get_object("sample-data", "data/application_test.csv")
        df_all = pd.read_csv(BytesIO(response.read()))

        row = df_all[df_all["SK_ID_CURR"] == sk_id]
        if row.empty:
            return {"error": f"ID {sk_id} not found"}
        return self.predictor.inference(row)


# ----------------------------------------------------------------------
# 4) FastAPI app
# ----------------------------------------------------------------------
cfg = ApiConfig()  # Predictor sẽ tự gọi cfg.configure_mlflow()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = await PredictionService.create(cfg)
    yield


app = FastAPI(lifespan=lifespan)


def get_service() -> PredictionService:
    return app.state.service


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/Prediction")
def predict(
    items: List[RawItem] = Body(...),
    service: PredictionService = Depends(get_service),
):
    return service.predict_items(items)


@app.post("/Prediction-by-id")
def predict_by_id(id: int, service: PredictionService = Depends(get_service)):
    return service.predict_by_id(id)
