import sys
import types
from pathlib import Path
import pytest
import random
from faker import Faker
import os
import pandas as pd
import numpy as np
import joblib
import json
from io import BytesIO
from typing import Union, get_origin, get_args
import contextlib, types, uuid
from types import SimpleNamespace, ModuleType


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from client.api.schema import RawItem

fake = Faker()


# ---------------- Modules patched -------------------
"""Patch Minio and MLflow modules for testing.
These class are dummies implementations which patch the original Minio and MLflow clients from client module. 

In these classes, I patched all args and kwargs to avoid any issues with the original client. These args and kwargs can be found in original Minio and Mlflow documentation.
"""


class FakeMinio:
    def __init__(self, *args, **kwargs):
        pass

    def fget_object(self, bucket, key, dest):
        Path(dest).write_text(f"dummy {bucket}/{key}")

    def fput_object(self, bucket, key, src):
        Path(src).read_bytes()

    def get_object(self, bucket, key):
        if bucket == "sample-data" and key == "data/application_test.csv":
            path = ROOT / "src" / "client" / "joblib" / "application_test.csv"
            return BytesIO(path.read_bytes())
        return BytesIO(b"")

    def put_object(self, bucket, key, data, length, *args, **kwargs):
        return


class DummyVersion:
    def __init__(self, version="1", stage="Production"):
        self.version = version
        self.current_stage = stage


class DummyMlflowClient:
    def get_latest_versions(self, model_name, stages=None):
        return [DummyVersion(version="123", stage="Production")]

    def download_artifacts(self, run_id, path, dst_path):
        return str(ROOT / "src" / "client" / "joblib" / "transformer.joblib")


class FakeMlflow(types.ModuleType):
    def __init__(self):
        super().__init__("mlflow")

        def _dummy_run():
            info = SimpleNamespace(run_id=str(uuid.uuid4()))
            return SimpleNamespace(info=info)

        class _DummyRegistry:
            def __init__(self, name, version="1"):
                self.name = name
                self.version = version

        # -------- sub-modules ----- #
        xgb = ModuleType("mlflow.xgboost")
        xgb.log_model = lambda *args, **kwargs: None
        xgb.load_model = lambda *args, **kwargs: joblib.load(
            str(ROOT / "src" / "client" / "joblib" / "model.joblib")
        )

        self.xgboost = xgb

        lgb = ModuleType("mlflow.lightgbm")
        lgb.log_model = lambda *args, **kwargs: None
        lgb.load_model = lambda *args, **kwargs: joblib.load(
            str(ROOT / "src" / "client" / "joblib" / "model.joblib")
        )

        self.lightgbm = lgb

        tracking = ModuleType("mlflow.tracking")
        tracking.MlflowClient = lambda *a, **k: DummyMlflowClient()
        self.tracking = tracking

        # -------- core API -------- #
        self.set_tracking_uri = lambda *args, **kwargs: None
        self.set_experiment = lambda *args, **kwargs: None
        self.start_run = lambda *args, **kwargs: contextlib.nullcontext(_dummy_run())
        self.end_run = lambda *args, **kwargs: None
        self.active_run = lambda *args, **kwargs: None

        self.log_params = lambda *args, **kwargs: None
        self.log_metric = lambda *args, **kwargs: None
        self.log_artifact = lambda *args, **kwargs: None
        self.log_artifacts = lambda *args, **kwargs: None
        self.get_artifact_uri = lambda *args, **kwargs: "file://dummy"

        self.register_model = lambda model_uri, name: _DummyRegistry(
            name=name, version="1"
        )


@pytest.fixture(scope="session", autouse=True)
def patch_minio_and_mlflow():
    sys.modules["mlflow"] = fake_mlflow = FakeMlflow()
    sys.modules["mlflow.tracking"] = fake_mlflow.tracking
    sys.modules["mlflow.xgboost"] = fake_mlflow.xgboost
    sys.modules["mlflow.lightgbm"] = fake_mlflow.lightgbm
    sys.modules["minio"] = types.SimpleNamespace(Minio=FakeMinio)

    os.environ["PARENT_RUN_ID"] = "1234567890abcdef1234567890abcdef"
    os.environ["TRANSFORMER_ARTIFACT_PATH"] = "transformer.joblib"
    os.environ["MODEL_NAME"] = "fake_model"
    os.environ["MODEL_TYPE"] = "xgb"


@pytest.fixture(scope="session", autouse=True)
def exclude_opentelemetry_imports():
    fake_metrics = types.SimpleNamespace(
        set_meter_provider=lambda *args, **kwargs: None,
        get_meter_provider=lambda: types.SimpleNamespace(
            get_meter=lambda name: types.SimpleNamespace(
                create_observable_gauge=lambda *args, **kwargs: None
            )
        ),
        Observation=lambda v: v,
    )
    fake_sdk_metrics = types.SimpleNamespace(MeterProvider=lambda *args, **kwargs: None)
    fake_exporter_prometheus = types.SimpleNamespace(
        PrometheusMetricReader=lambda *args, **kwargs: None,
        start_http_server=lambda *args, **kwargs: None,
    )
    fake_prometheus_client = types.SimpleNamespace(
        start_http_server=lambda *args, **kwargs: None
    )

    sys.modules["opentelemetry"] = types.SimpleNamespace(metrics=fake_metrics)
    sys.modules["opentelemetry.metrics"] = fake_metrics
    sys.modules["opentelemetry.sdk"] = types.SimpleNamespace(metrics=fake_sdk_metrics)
    sys.modules["opentelemetry.sdk.metrics"] = fake_sdk_metrics
    sys.modules["opentelemetry.exporter.prometheus"] = fake_exporter_prometheus

    sys.modules["prometheus_client"] = fake_prometheus_client


# --------------- API conftest -------------------


@pytest.fixture
def sample_payload():
    payload_file = Path(__file__).parent / "sample_payload.json"
    return json.loads(payload_file.read_text())


def generate_fake_value(origin_payload_type):
    if origin_payload_type == int:
        return random.randint(0, 100)
    if origin_payload_type == float:
        return round(random.uniform(0.0, 10000.0), 2)
    if origin_payload_type == str:
        return fake.word()
    return None


def build_fake_rawitem_dict() -> dict:
    data = {}
    for field, ann in RawItem.__annotations__.items():
        origin, args = get_origin(ann), get_args(ann)
        origin_payload_type = args[0] if origin is Union and type(None) in args else ann
        val = generate_fake_value(origin_payload_type)
        if val is not None:
            data[field] = val
    return data


@pytest.fixture
def fake_csv(tmp_path: Path):
    rows = [build_fake_rawitem_dict() for _ in range(10)]
    df = pd.DataFrame(rows)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError(" No usable numeric features generated!")

    df["TARGET"] = [0, 1] * (len(df) // 2) + [0] * (len(df) % 2)

    dst = tmp_path / "fake.csv"
    df.to_csv(dst, index=False)
    return dst


@pytest.fixture
def expected_result():
    return {
        "predictions": [
            {
                "result": "Accept",
                "prob_accept": 0.9986466765403748,
                "prob_decline": 0.0013533371966332197,
                "entropy": -0.0148,
                "confidence": 0.9986,
            }
        ],
        "metrics": {
            "avg_entropy": -0.01484741736203432,
            "avg_confidence": 0.9986466765403748,
        },
    }


@pytest.fixture
def strip_time_field():
    def strip(body: dict) -> dict:
        if "inference_time_ms" in body:
            body = dict(body)  # copy
            body.pop("inference_time_ms")
        return body

    return strip


# add this fixture to for using test other optuna model hyperparam tuning if you want to test another model, in my case, I'm testing registered model
@pytest.fixture
def compare_results():
    def compare(actual, expected, decimal=3):
        for key in expected:
            if isinstance(expected[key], dict):
                compare(actual[key], expected[key], decimal=decimal)
            elif isinstance(expected[key], list):
                assert len(actual[key]) == len(expected[key])
                for a, e in zip(actual[key], expected[key]):
                    compare(a, e, decimal=decimal)
            elif isinstance(expected[key], float):
                import pytest

                assert actual[key] == pytest.approx(expected[key], rel=10**-decimal)
            else:
                assert actual[key] == expected[key]

    return compare
