from contextlib import asynccontextmanager
from functools import wraps
from io import BytesIO
import asyncio
from time import time
from typing import List, Dict, Any
import numpy as np
import pandas as pd

from fastapi import FastAPI, Body, Depends
from fastapi.responses import JSONResponse

from opentelemetry import metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from prometheus_client import start_http_server

from schema import RawItem
from utils import map_evidently_data, custom_evidently_report, ApiConfig, Predictor


def entropy(p: np.ndarray) -> float:
    return float(np.sum(p * np.log2(p + 1e-10)))


def confidence(p: np.ndarray) -> float:
    return float(p.max())


class MetricsHandler:
    def __init__(self):
        self.avg_entropy = 0.0
        self.avg_confidence = 0.0
        self.model_accuracy = 0.0
        self.roc_auc = 0.0
        self.drift_score = 0.0
        self.missing_value_count = 0.0
        self.duplicated_row_count = 0.0

        # Init Prometheus
        reader = PrometheusMetricReader()
        metrics.set_meter_provider(MeterProvider(metric_readers=[reader]))
        meter = metrics.get_meter_provider().get_meter("prediction_api")
        start_http_server(addr="0.0.0.0", port=8001)

        # Đăng ký các observable gauge metric
        meter.create_observable_gauge(
            "api_avg_entropy",
            callbacks=[lambda opts: [metrics.Observation(self.avg_entropy)]],
        )
        meter.create_observable_gauge(
            "api_avg_confidence",
            callbacks=[lambda opts: [metrics.Observation(self.avg_confidence)]],
        )
        meter.create_observable_gauge(
            "api_model_accuracy",
            callbacks=[lambda opts: [metrics.Observation(self.model_accuracy)]],
        )
        meter.create_observable_gauge(
            "api_roc_auc",
            callbacks=[lambda opts: [metrics.Observation(self.roc_auc)]],
        )
        meter.create_observable_gauge(
            "api_drift_score",
            callbacks=[lambda opts: [metrics.Observation(self.drift_score)]],
        )
        meter.create_observable_gauge(
            "api_missing_value_count",
            callbacks=[lambda opts: [metrics.Observation(self.missing_value_count)]],
        )
        meter.create_observable_gauge(
            "api_duplicated_row_count",
            callbacks=[lambda opts: [metrics.Observation(self.duplicated_row_count)]],
        )

    def update(self, ents, confs):
        self.avg_entropy = float(np.mean(ents)) if ents else 0.0
        self.avg_confidence = float(np.mean(confs)) if confs else 0.0

    def update_from_evidently(self, report_dict):
        def get_nested(dct, keys, default=0.0):
            for k in keys:
                if isinstance(dct, dict) and k in dct:
                    dct = dct[k]
                else:
                    return default
            return dct

        # Map các metrics lấy từ report Evidently
        self.model_accuracy = get_nested(
            report_dict, ["metrics", 0, "result", "accuracy"]
        )
        self.roc_auc = get_nested(report_dict, ["metrics", 1, "result", "roc_auc"])
        self.drift_score = get_nested(
            report_dict, ["metrics", 3, "result", "dataset_drift", "drift_score"]
        )
        self.missing_value_count = get_nested(
            report_dict, ["metrics", 4, "result", "current"]
        )
        self.duplicated_row_count = get_nested(
            report_dict, ["metrics", 5, "result", "current"]
        )


# ----------------------------------------------------------------------
"""
Create a decorator to handle OpenTelemetry metrics, for further metrics collection you can modify 
- MetricHandler class: define gauge and counter
- otel_metric decorator: define object to be collected by OpenTelemetry in the POST request of the API 
"""


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
""" 
The main class for to creating the prediction service, which initializes the predictor and handles prediction requests.
This will create a new instance of `Predictor` with provided configuration from `ApiConfig`
The asynchonous function `create` is used to ensure Predictor is loaded in a separate thread, allowing the FastAPI app to start without blocking.
"""


# ----------------------------------------------------------------------
class PredictionService:
    def __init__(self, cfg: ApiConfig, predictor: Predictor):
        self.cfg = cfg
        self.predictor = predictor
        self.metrics = MetricsHandler()

    @classmethod
    def create(cls, cfg: ApiConfig):
        predictor = Predictor(cfg)
        predictor.load_artifacts()
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
                "avg_entropy": self.metrics.avg_entropy,
                "avg_confidence": self.metrics.avg_confidence,
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
"""
The main FastAPI app which initializes the PredictionService and defines the API endpoints.

The `contextmanager` is used to manage the lifespan the app, ensure PredictionService is available before handling requests.
"""


# ----------------------------------------------------------------------
def create_app():
    cfg = ApiConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = PredictionService.create(cfg)
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


    @app.get("/data-monitor")
    def data_monitor(service: PredictionService = Depends(get_service)):
        config = service.cfg
        reference_data, current_data = map_evidently_data(config)
        report = custom_evidently_report(reference_data, current_data)
        report_dict = report.as_dict()
        service.metrics.update_from_evidently(report_dict)  
        return JSONResponse(content=report_dict)

    return app


app = create_app()
