from io import BytesIO
from time import time
from typing import List
import os

import numpy as np
import pandas as pd

from fastapi import FastAPI, Body, Depends
from evidently.ui.workspace import RemoteWorkspace

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import get_tracer_provider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource

from client.api.metrics_handler import MetricsHandler, otel_metric
from client.api.schema import RawItem
from client.api.utils import (
    map_evidently_data,
    custom_evidently_report,
    ApiConfig,
    Predictor,
)

os.getenv("JAEGER_AGENT_HOST")

trace.set_tracer_provider(
    TracerProvider(resource=Resource.create({SERVICE_NAME: "prediction_api_service"}))
)

tracer = get_tracer_provider().get_tracer("prediction_api", "0.1")

jaeger_exporter = JaegerExporter(
    agent_host_name=os.getenv("JAEGER_AGENT_HOST"),
    agent_port=6831,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
get_tracer_provider().add_span_processor(span_processor)

# ----------------------------------------------------------------------
""" 
The main class for to creating the prediction service, which initializes the predictor and handles prediction requests.
This will create a new instance of `Predictor` with provided configuration from `ApiConfig`
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
    ):
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
        with tracer.start_as_current_span("predict_items"):
            df = pd.DataFrame([i.dict() for i in items]).replace({None: np.nan})
            return self.predictor.inference(df)

    @otel_metric
    def predict_by_id(self, sk_id: int):
        with tracer.start_as_current_span("predict_by_id") as span:
            with tracer.start_as_current_span(
                "data-loader", links=[trace.Link(span.get_span_context())]
            ):
                minio_client = self.cfg.get_minio_client()
                response = minio_client.get_object(
                    "sample-data", "data/application_test.csv"
                )
                df_all = pd.read_csv(BytesIO(response.read()))

                row = df_all[df_all["SK_ID_CURR"] == sk_id]
                if row.empty:
                    return {"error": f"ID {sk_id} not found"}
            return self.predictor.inference(row)


# ----------------------------------------------------------------------
"""
The main FastAPI app which initializes the PredictionService and defines the API endpoints.

"""


# ----------------------------------------------------------------------
cfg = ApiConfig()
prediction_service = PredictionService.create(cfg)


def get_service() -> PredictionService:
    return prediction_service


app = FastAPI()


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/prediction")
def predict(
    items: List[RawItem] = Body(...),
    service: PredictionService = Depends(get_service),
):
    return service.predict_items(items)


@app.post("/prediction-by-id")
def predict_by_id(id: int, service: PredictionService = Depends(get_service)):
    return service.predict_by_id(id)


@app.get("/data-monitor")
def data_monitor(service: PredictionService = Depends(get_service)):
    with tracer.start_as_current_span("data_monitor") as span:
        with tracer.start_as_current_span(
            "data-drift-loader", links=[trace.Link(span.get_span_context())]
        ):
            cfg = service.cfg
            workspace_path = cfg.evidently_workspace
            evidently_ws = RemoteWorkspace(workspace_path)

            project_name = "credit_underwriting"
            project = None
            existing = evidently_ws.search_project(project_name)
            if existing:
                project = existing[0]
            else:
                project = evidently_ws.create_project(project_name)
                project.description = "Monitoring credit underwriting snapshots"
                project.save()

            reference_data, current_data = map_evidently_data(cfg)
            snapshot = custom_evidently_report(reference_data, current_data)

        with tracer.start_as_current_span(
            "evidently-report", links=[trace.Link(span.get_span_context())]
        ):
            evidently_ws.add_run(project.id, snapshot)

        return {
                "status": "stored",
                "project_id": project.id,
                "project_name": project_name,
            }

