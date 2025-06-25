import sys
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
import os

from tests.mock_utils import build_mock_mlflow
from tests.test_utils import DummyBinningProcess, DummySelector, FakeMinioResponse


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
def patch_env(mocker, dummy_joblib_path, monkeypatch):
    mlflow = build_mock_mlflow()
    monkeypatch.setitem(sys.modules, "mlflow", mlflow)

    mlflow.MlflowClient = mlflow.tracking.MlflowClient
    monkeypatch.setitem(sys.modules, "mlflow.tracking", mlflow)

    mocker.patch("src.pipeline.scripts.component_utils.mlflow", mlflow)
    mocker.patch("src.client.api.utils.MlflowClient",mlflow.tracking.MlflowClient)

    mlflow.tracking.MlflowClient.return_value.download_artifacts.return_value = str(dummy_joblib_path)
    mocker.patch("prometheus_client.start_http_server", return_value=None)

    monkeypatch.setenv("MODEL_TYPE", "xgb")
    monkeypatch.setenv("MODEL_NAME", "dummy")
    monkeypatch.setenv("S3_ENDPOINT", "fake:9000")
    monkeypatch.setenv("S3_ACCESS_KEY", "abc")
    monkeypatch.setenv("S3_SECRET_KEY", "abc")
    monkeypatch.setenv("MLFLOW_ENDPOINT", "fake:1234")
    monkeypatch.setenv("EVIDENTLY_WORKSPACE", "http://fake-evidently:8000")
    monkeypatch.setenv("PREDICTION_API_URL", "http://mocked-api")


@pytest.fixture
def patch_minio(mocker):
    csv_bytes = b"SK_ID_CURR,xyz\n100001,456\n"
    fake_response = FakeMinioResponse(csv_bytes)

    minio_mock = mocker.MagicMock()
    minio_mock.get_object.return_value = fake_response

    mocker.patch("src.client.api.main.cfg.get_minio_client", return_value=minio_mock)
    return minio_mock


@pytest.fixture(scope="session")
def dummy_joblib_path():
    tmpdir = tempfile.mkdtemp()
    joblib.dump(
        DummyBinningProcess(), os.path.join(tmpdir, "opt_binning_process.joblib")
    )
    joblib.dump(DummySelector(), os.path.join(tmpdir, "feat_selector.joblib"))
    yield tmpdir
    import shutil

    shutil.rmtree(tmpdir)
