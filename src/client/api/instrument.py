from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from prometheus_client import start_http_server
from opentelemetry.metrics import set_meter_provider

import os

JAEGER_AGENT_HOST = os.getenv("JAEGER_AGENT_HOST")


def setup_tracer(
    service_name="prediction_api_service",
    jaeger_host=JAEGER_AGENT_HOST,
    jaeger_port=6831,
):
    resource = Resource.create({SERVICE_NAME: service_name})
    trace.set_tracer_provider(TracerProvider(resource=resource))
    tracer = trace.get_tracer_provider().get_tracer("prediction_api", "0.1")
    jaeger_exporter = JaegerExporter(
        agent_host_name=jaeger_host, agent_port=jaeger_port
    )
    span_processor = BatchSpanProcessor(jaeger_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)
    return tracer


def setup_meter(
    service_name="prediction-service", prometheus_port=8001, prometheus_addr="0.0.0.0"
):
    resource = Resource.create({SERVICE_NAME: service_name})
    start_http_server(port=prometheus_port, addr=prometheus_addr)
    reader = PrometheusMetricReader()
    provider = MeterProvider(resource=resource, metric_readers=[reader])
    set_meter_provider(provider)
    meter = metrics.get_meter("prediction_spread", "0.1.1")
    return meter


def setup_metrics(meter):
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
    return prediction_counter, prediction_latency, error_counter, batch_size_hist