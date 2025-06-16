from opentelemetry import metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from prometheus_client import start_http_server
from typing import List
from functools import wraps

import numpy as np
from time import time
def entropy(p: np.ndarray):
    return float(np.sum(p * np.log2(p + 1e-10)))


def confidence(p: np.ndarray):
    return float(p.max())


class MetricsHandler:
    def __init__(self):
        self.avg_entropy = 0.0
        self.avg_confidence = 0.0

        reader = PrometheusMetricReader()
        metrics.set_meter_provider(MeterProvider(metric_readers=[reader]))
        meter = metrics.get_meter_provider().get_meter("prediction_api")
        start_http_server(addr="0.0.0.0", port=8001)

        meter.create_observable_gauge(
            "api_prediction_entropy",
            callbacks=[lambda _opts: [metrics.Observation(self.avg_entropy)]],
        )
        meter.create_observable_gauge(
            "api_avg_confidence",
            callbacks=[lambda _opts: [metrics.Observation(self.avg_confidence)]],
        )

        self.prediction_counter = meter.create_counter(
            "api_prediction_count",
            description="Count of predictions made by the API",
        )

        self.prediction_latency = meter.create_histogram(
            "api_prediction_latency",
            description="Latency of predictions made by the API",
            unit="ms",
        )

    def update(self, ents: List[float], confs: List[float]):
        self._avg_entropy = float(np.mean(ents))
        self._avg_confidence = float(np.mean(confs))


# ----------------------------------------------------------------------
"""
Create a decorator to handle OpenTelemetry metrics, for further metrics collection you can modify 
- MetricHandler class: define gauge and counter
- otel_metric decorator: define object to be collected by OpenTelemetry in the POST request of the API 
"""


# ----------------------------------------------------------------------
def otel_metric(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        start_time = time()
        result = fn(self, *args, **kwargs)
        latency_ms = (time() - start_time) * 1000

        preds, proba = result
        num_preds = len(preds)

        self.metrics.prediction_counter.add(num_preds)
        self.metrics.prediction_latency.record(latency_ms)

        ents = [entropy(p) for p in proba]
        confs = [confidence(p) for p in proba]
        self.metrics.update(ents, confs)
        return self._build_response(start_time, preds, proba, ents, confs)


    return wrapper

