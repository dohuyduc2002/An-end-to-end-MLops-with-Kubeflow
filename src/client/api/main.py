from time import time
from typing import List
import os
from functools import wraps
from io import BytesIO
import numpy as np
import pandas as pd

from fastapi import FastAPI, Body
from evidently.ui.workspace import RemoteWorkspace

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import get_tracer_provider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry import metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from prometheus_client import start_http_server

from client.api.schema import RawItem
from utils import (
    map_evidently_data,
    custom_evidently_report,
    ApiConfig,
    load_artifacts,
    entropy,
    confidence
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


reader = PrometheusMetricReader()
metrics.set_meter_provider(MeterProvider(metric_readers=[reader]))
meter = metrics.get_meter_provider().get_meter("prediction_api")
start_http_server(addr="0.0.0.0", port=8001)

prediction_counter = meter.create_counter(
    "api_prediction_count",
    description="Count of predictions made by the API",
)

prediction_latency = meter.create_histogram(
    "api_prediction_latency",
    description="Latency of predictions made by the API",
    unit="ms",
)

error_counter = meter.create_counter(
    "api_prediction_error_count",
    description="Number of failed prediction requests",
)

batch_size_hist = meter.create_histogram(
    "api_prediction_batch_size",
    description="Batch size of prediction requests",
)


def otel_metric(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        start_time = time()
        try:
            response = fn(*args, **kwargs)
        except Exception as e:
            error_counter.add(1)
            raise
        latency_ms = (time() - start_time) * 1000

        batch_size = len(response["predictions"])
        prediction_counter.add(batch_size)
        prediction_latency.record(latency_ms)
        batch_size_hist.record(batch_size)
        return response

    return wrapper


# ----------------------------------------------------------------------
cfg = ApiConfig()
app = FastAPI()


@app.get("/")
def health():
    return {"status": "ok"}


@otel_metric
@app.post("/prediction")
def predict(self, items: List[RawItem] = Body(...)):
    with tracer.start_as_current_span("predict_items"):
        start_ts = time()

        binning, selector, model = load_artifacts(cfg)

        df = pd.DataFrame([i.model_dump() for i in items]).replace({None: np.nan})
        df = df[[c for c in df.columns if c in binning.variable_names]]

        X = selector.transform(binning.transform(df))
        proba = model.predict_proba(X)
        preds = proba.argmax(axis=1)

        entropies = [entropy(p) for p in proba]
        confidences = [confidence(p) for p in proba]

        return {
            "prediction_method": "batch",
            "inference_time_ms": round((time() - start_ts) * 1000, 2),
            "predictions": [
                {
                    "result": "Accept" if y == 0 else "Decline",
                    "prob_accept": float(p[0]),
                    "prob_decline": float(p[1]),
                    "entropy": round(e, 4),
                    "confidence": round(c, 4),
                }
                for y, p, e, c in zip(preds, proba, entropies, confidences)
            ]
        }


@otel_metric
@app.post("/prediction-by-id")
def predict_by_id(id: int):
    with tracer.start_as_current_span("predict_items"):
        start_ts = time()

        binning, selector, model = load_artifacts(cfg)

        minio_client = cfg.get_minio_client()
        response = minio_client.get_object(
                    "sample-data", "data/application_test.csv"
                )
        feature_df = pd.read_csv(BytesIO(response.read()))

        X = selector.transform(binning.transform(feature_df))
        proba = model.predict_proba(X)
        preds = proba.argmax(axis=1)

        entropy_single = entropy(proba)
        confidence_single = confidence(proba)

        return {
            "prediction_method": "single",
            "inference_time_ms": round((time() - start_ts) * 1000, 2),
            "predictions": [
                {
                    "result": "Accept" if preds == 0 else "Decline",
                    "prob_accept": proba[0][0],
                    "prob_decline": proba[0][1],
                    "entropy": entropy_single,
                    "confidence": confidence_single,
                }
            ]
        }


@app.get("/data-monitor")
def data_monitor():
    with tracer.start_as_current_span("data_monitor") as span:
        with tracer.start_as_current_span(
            "data-drift-loader", links=[trace.Link(span.get_span_context())]
        ):
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
