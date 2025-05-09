# testing/conftest.py

import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv
from minio import Minio
import mlflow

# ─── Add PROJECT_ROOT/src to sys.path ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

# ─── Load .env file (if present) ─────────────────────────────────────────────
dotenv_path = PROJECT_ROOT / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path, override=False)

# ─── Override DNS for containerized test (e.g. Jenkins or docker run) ────────
if os.getenv("PYTEST_DOCKER") == "1":
    os.environ["MINIO_ENDPOINT"] = "minio.minio.svc.cluster.local:9000"
    os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow.mlflow.svc.cluster.local:5000"

# ─── Fixtures ────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def minio_client():
    return Minio(
        endpoint=os.environ["MINIO_ENDPOINT"],
        access_key=os.environ["MINIO_ACCESS_KEY"],
        secret_key=os.environ["MINIO_SECRET_KEY"],
        secure=False,
    )

@pytest.fixture(scope="session")
def mlflow_client():
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    return mlflow.tracking.MlflowClient()

@pytest.fixture(scope="session")
def sample_data(minio_client):
    bucket = os.getenv("MINIO_BUCKET_NAME", "sample-data")
    tr_key, te_key = "data/application_train.csv", "data/application_test.csv"
    minio_client.stat_object(bucket, tr_key)
    minio_client.stat_object(bucket, te_key)
    return bucket, tr_key, te_key
