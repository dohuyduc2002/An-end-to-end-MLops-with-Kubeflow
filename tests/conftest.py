import sys
import types
from pathlib import Path
import pytest
import random
from faker import Faker
import pandas as pd
import numpy as np
import json
import tempfile
import joblib
from typing import Union, get_origin, get_args
import types
from io import BytesIO

from tests.mock_utils import (
    build_mock_mlflow,
    build_mock_optuna
)
from tests.test_utils import DummyBinningProcess, DummySelector


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from client.api.schema import RawItem
fake = Faker()


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


def build_fake_rawitem_dict():
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
def patch_env_kfp(monkeypatch):
    import sys
    import optuna

    mock_mlflow = build_mock_mlflow()
    mock_optuna = build_mock_optuna()

    monkeypatch.setitem(sys.modules, "mlflow", mock_mlflow)
    monkeypatch.setattr(optuna, "create_study", lambda direction: mock_optuna)
    
    yield mock_mlflow


@pytest.fixture
def patch_env_api(mocker, dummy_joblib_path, monkeypatch):
    mock_mlflow = build_mock_mlflow()

    # Patch sys.modules cho mlflow
    mocker.patch.dict("sys.modules", {"mlflow": mock_mlflow})

    tracking_mod = types.ModuleType("mlflow.tracking")
    tracking_mod.MlflowClient = mock_mlflow.tracking.MlflowClient
    mocker.patch.dict("sys.modules", {"mlflow.tracking": tracking_mod})

    client_instance = mock_mlflow.tracking.MlflowClient.return_value
    client_instance.download_artifacts.return_value = str(dummy_joblib_path)

    minio_mock = mocker.MagicMock()
    minio_mock.get_object.side_effect = lambda bucket, object_name: BytesIO(
        b"SK_ID_CURR,xyz\n100001,456\n"  # sample csv byte data
    )
    mocker.patch("minio.Minio", return_value=minio_mock)

    mocker.patch("prometheus_client.start_http_server", return_value=None)

    # Patch env cho config
    monkeypatch.setenv("MODEL_TYPE", "xgb")
    monkeypatch.setenv("MODEL_NAME", "dummy")
    monkeypatch.setenv("S3_ENDPOINT", "fake:9000")
    monkeypatch.setenv("S3_ACCESS_KEY", "abc")
    monkeypatch.setenv("S3_SECRET_KEY", "abc")
    monkeypatch.setenv("MLFLOW_ENDPOINT", "fake:5000")
    monkeypatch.setenv("EVIDENTLY_WORKSPACE", "/tmp")


@pytest.fixture(scope="session")
def dummy_joblib_path():
    obj = {
        "opt_binning_process": DummyBinningProcess(),
        "selector": DummySelector(),
    }
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        joblib.dump(obj, f)
        return f.name  


@pytest.fixture
def sample_result():
    return {
        "predictions": [
            {
                "result": "abc",
                "prob_accept": 1,
                "prob_decline": 1,
                "entropy": 1,
                "confidence": 1,
            }
        ],
        "metrics": {
            "avg_entropy": 1,
            "avg_confidence": 1,
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


@pytest.fixture
def compare_schema():
    def compare(actual, expected):
        for key in expected:
            assert key in actual
            if isinstance(expected[key], dict):
                compare(actual[key], expected[key])
            else:
                pass
    return compare
