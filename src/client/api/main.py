from time import time
from typing import List
from functools import wraps
from io import BytesIO
import numpy as np
import pandas as pd

from client.api.schema import RawItem
from client.api.utils import (
    map_evidently_data,
    custom_evidently_report,
    ApiConfig,
    load_artifacts,
)
from client.api.instrument import setup_tracer, setup_meter, setup_metrics

from fastapi import FastAPI, Body, HTTPException, Request
from evidently.ui.workspace import RemoteWorkspace
from opentelemetry import trace

tracer = setup_tracer()
meter = setup_meter()
prediction_counter, prediction_latency, error_counter, batch_size_hist = setup_metrics(
    meter
)


def otel_metric(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        start_time = time()
        try:
            response = fn(*args, **kwargs)
        except Exception:
            error_counter.add(1)
            raise
        latency_ms = (time() - start_time) * 1000
        batch_size = len(response["predictions"])
        prediction_counter.add(batch_size)
        prediction_latency.record(latency_ms)
        batch_size_hist.record(batch_size)
        return response

    return wrapper


# Prediction logic
def predict_batch(items: List[RawItem], binning, selector, model):
    df = pd.DataFrame([i.model_dump() for i in items]).replace({None: np.nan})
    df = df[[c for c in df.columns if c in binning.variable_names]]
    X = selector.transform(binning.transform(df))
    proba = model.predict_proba(X)
    preds = proba.argmax(axis=1)
    result = [
        {
            "result": "Accept" if y == 0 else "Decline",
            "prob_accept": float(p[0]),
            "prob_decline": float(p[1]),
        }
        for y, p in zip(preds, proba)
    ]
    return result


def predict_by_id(
    id: int,
    minio_client,
    binning,
    selector,
    model,
    bucket: str = "sample-data",
    key: str = "data/application_test.csv",
):
    response = minio_client.get_object(bucket, key)
    feature_df = pd.read_csv(BytesIO(response.read()))
    row = feature_df[feature_df["SK_ID_CURR"] == id]
    if row.empty:
        raise HTTPException(status_code=404, detail="ID not found")
    X = selector.transform(binning.transform(row))
    proba = model.predict_proba(X)
    preds = proba.argmax(axis=1)
    return [
        {
            "result": "Accept" if preds[0] == 0 else "Decline",
            "prob_accept": float(proba[0][0]),
            "prob_decline": float(proba[0][1]),
        }
    ]


# lifespan function to load artifacts and initialize the app state
def lifespan(app: FastAPI):
    cfg = ApiConfig()
    binning, selector, model = load_artifacts(cfg)
    app.state.binning = binning
    app.state.selector = selector
    app.state.model = model
    yield


def create_app(lifespan=lifespan):
    app = FastAPI(lifespan=lifespan)

    @app.get("/")
    def health():
        return {"status": "ok"}

    @app.post("/prediction")
    @otel_metric
    def prediction(request: Request, items: List[RawItem] = Body(...)):
        with tracer.start_as_current_span("predict_items"):
            start_ts = time()
            binning = request.app.state.binning
            selector = request.app.state.selector
            model = request.app.state.model
            preds = predict_batch(items, binning, selector, model)
            return {
                "prediction_method": "batch",
                "inference_time_ms": round((time() - start_ts) * 1000, 2),
                "predictions": preds,
            }

    @app.post("/prediction-by-id")
    @otel_metric
    def prediction_by_id_route(request: Request, id: int):
        with tracer.start_as_current_span("predict_items") as span:
            start_ts = time()
            binning = request.app.state.binning
            selector = request.app.state.selector
            model = request.app.state.model
            with tracer.start_as_current_span(
                "fetch_data_from_minio", links=[trace.Link(span.get_span_context())]
            ):
                cfg = ApiConfig()
                minio_client = cfg.get_minio_client()
                try:
                    preds = predict_by_id(
                        id,
                        minio_client,
                        binning,
                        selector,
                        model,
                        bucket="sample-data",
                        key="data/application_test.csv",
                    )
                except HTTPException as e:
                    raise e
            return {
                "prediction_method": "single",
                "inference_time_ms": round((time() - start_ts) * 1000, 2),
                "predictions": preds,
            }

    @app.get("/data-monitor")
    def data_monitor(request: Request):
        with tracer.start_as_current_span("data_monitor") as span:
            with tracer.start_as_current_span(
                "data-drift-loader", links=[trace.Link(span.get_span_context())]
            ):
                cfg = ApiConfig()
                workspace_path = cfg.evidently_workspace
                evidently_ws = RemoteWorkspace(workspace_path)
                project_name = cfg.evidently_project_name
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

    return app