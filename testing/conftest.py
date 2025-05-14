import sys
import pytest
import pandas as pd
from pathlib import Path
from typing import get_args, get_origin, Union
from unittest.mock import MagicMock
from faker import Faker

# ─── Add src/ to sys.path ───────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_PATH))

# ─── Register custom ini option ─────────────────────────────────────────
def pytest_addoption(parser):
    parser.addini("test_image", "Docker image used for KFP components", default="microwave1005/kfp-ci-jenkins:latest")

# ─── Fixtures ───────────────────────────────────────────────────────────
from client.app.schema import RawItem
fake = Faker()

def build_fake_value(field_type):
    if field_type == int:
        return fake.random_int(min=0, max=100)
    elif field_type == float:
        return fake.pyfloat(left_digits=5, right_digits=4, positive=True)
    return None  # skip str

def build_fake_raw_item() -> RawItem:
    kwargs = {}
    for field_name, annotation in RawItem.__annotations__.items():
        origin = get_origin(annotation)
        args = get_args(annotation)
        actual_type = args[0] if origin is Union and type(None) in args else annotation
        if actual_type in [int, float]:
            kwargs[field_name] = build_fake_value(actual_type)
    return RawItem(**kwargs)

@pytest.fixture
def fake_raw_items():
    return [build_fake_raw_item() for _ in range(50)]

@pytest.fixture
def mock_minio_client(fake_raw_items):
    df = pd.DataFrame([item.dict() for item in fake_raw_items])
    df_train = df.copy()
    df_test = df.drop(columns=["TARGET"], errors="ignore")

    output_paths = {}

    def fget(bucket, key, dest_path):
        content_df = df_train if "train" in key else df_test
        content_df.to_csv(dest_path, index=False)
        output_paths[key] = pd.read_csv(dest_path)

    def fput(bucket, key, src_path):
        assert Path(src_path).exists()

    client = MagicMock()
    client.fget_object.side_effect = fget
    client.fput_object.side_effect = fput
    client.output_paths = output_paths
    return client

@pytest.fixture
def mock_mlflow_client():
    client = MagicMock()
    run_ctx = MagicMock()
    client.start_run.return_value.__enter__.return_value = run_ctx
    client.get_artifact_uri.return_value = "s3://mock"
    return client

@pytest.fixture(scope="session")
def test_image(request):
    return request.config.getini("test_image")
