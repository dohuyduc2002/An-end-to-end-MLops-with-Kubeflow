import sys
import types
from pathlib import Path
import pytest
import random
import contextlib
from faker import Faker
import pandas as pd
import numpy as np
import joblib
import json
from io import BytesIO
from typing import Union, get_origin, get_args

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

fake = Faker()

class FakeMinio:
    def __init__(self, *_, **__): pass

    def fget_object(self, bucket, key, dest):
        Path(dest).write_text(f"dummy {bucket}/{key}")

    def fput_object(self, bucket, key, src):
        Path(src).read_bytes()
        
    def get_object(self, bucket, key):
        if bucket == "sample-data" and key == "data/application_test.csv":
            path = ROOT / "src" / "client" / "joblib" / "application_test.csv"
            return BytesIO(path.read_bytes())
        return BytesIO(b"")  

    def put_object(self, bucket, key, data, length, *_, **__):
        return

class DummyVersion:
    def __init__(self, version="1", stage="Production"):
        self.version = version
        self.current_stage = stage

class DummyMlflowClient:
    def get_latest_versions(self, model_name, stages=None):
        return [DummyVersion(version="123", stage="Production")]

class FakeMlflow(types.ModuleType):
    def __init__(self):
        super().__init__("mlflow")
        # mlflow.xgboost
        xgb = types.ModuleType("mlflow.xgboost")
        xgb.log_model = lambda *a, **k: None
        xgb.load_model = lambda *a, **k: "mock_xgb_model"
        self.xgboost = xgb
        # mlflow.lightgbm
        lgb = types.ModuleType("mlflow.lightgbm")
        lgb.log_model = lambda *a, **k: None
        lgb.load_model = lambda *a, **k: "mock_lgbm_model"
        self.lightgbm = lgb
        # mlflow.tracking
        tracking = types.ModuleType("mlflow.tracking")
        tracking.MlflowClient = lambda *a, **k: DummyMlflowClient()
        self.tracking = tracking
        # core
        self.set_tracking_uri = lambda *a, **k: None
        self.set_experiment = lambda *a, **k: None
        self.start_run = lambda *a, **k: contextlib.nullcontext()
        self.log_params = lambda *a, **k: None
        self.log_metric = lambda *a, **k: None
        self.log_artifacts = lambda *a, **k: None
        self.get_artifact_uri = lambda *a, **k: "file://dummy"
        self.register_model = lambda *a, **k: None


@pytest.fixture
def sample_payload():
    fp = Path(__file__).parent / "sample_payload.json"
    return json.loads(fp.read_text())

@pytest.fixture(scope="session", autouse=True)
def patch_minio_and_mlflow():
    sys.modules["mlflow"] = fake_mlflow = FakeMlflow()
    sys.modules["mlflow.xgboost"] = fake_mlflow.xgboost
    sys.modules["mlflow.lightgbm"] = fake_mlflow.lightgbm
    sys.modules["mlflow.tracking"] = fake_mlflow.tracking
    sys.modules["minio"] = types.SimpleNamespace(Minio=FakeMinio)

        
@pytest.fixture
def patch_api_joblib():
    import client.app.app as app  
    base_dir = ROOT / "src" / "client" / "joblib"
    app.model = joblib.load(base_dir / "model.joblib")
    app.transformer = joblib.load(base_dir / "transformer.joblib")

from client.app.schema import RawItem

def generate_fake_value(tp):
    if tp == int:
        return random.randint(0, 100)
    if tp == float:
        return round(random.uniform(0.0, 10000.0), 2)
    if tp == str:
        return fake.word()
    return None

def build_fake_rawitem_dict() -> dict:
    data = {}
    for field, ann in RawItem.__annotations__.items():
        origin, args = get_origin(ann), get_args(ann)
        real_type = args[0] if origin is Union and type(None) in args else ann
        val = generate_fake_value(real_type)
        if val is not None:
            data[field] = val
    return data

@pytest.fixture
def fake_csv(tmp_path: Path):
    rows = [build_fake_rawitem_dict() for _ in range(10)]
    df = pd.DataFrame(rows)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("❌ No usable numeric features generated!")

    df["TARGET"] = [0, 1] * (len(df) // 2) + [0] * (len(df) % 2)

    dst = tmp_path / "fake.csv"
    df.to_csv(dst, index=False)
    return dst

@pytest.fixture(scope="session", autouse=True)
def exclude_opentelemetry_imports():
    fake_metrics = types.SimpleNamespace(
        set_meter_provider=lambda *a, **k: None,
        get_meter_provider=lambda: types.SimpleNamespace(get_meter=lambda name: types.SimpleNamespace(
            create_observable_gauge=lambda *a, **k: None
        )),
        Observation=lambda v: v
    )
    fake_sdk_metrics = types.SimpleNamespace(
        MeterProvider=lambda *a, **k: None
    )
    fake_exporter_prometheus = types.SimpleNamespace(
        PrometheusMetricReader=lambda *a, **k: None,
        start_http_server=lambda *a, **k: None
    )

    sys.modules["opentelemetry"] = types.SimpleNamespace(metrics=fake_metrics)
    sys.modules["opentelemetry.metrics"] = fake_metrics
    sys.modules["opentelemetry.sdk"] = types.SimpleNamespace(metrics=fake_sdk_metrics)
    sys.modules["opentelemetry.sdk.metrics"] = fake_sdk_metrics
    sys.modules["opentelemetry.exporter.prometheus"] = fake_exporter_prometheus
