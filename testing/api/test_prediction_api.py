"""
Spin-up FastAPI app in-process & validate predictions.

FastAPI app file:  src/client/app/app.py   (IMPORT_PATH below)
"""

import os, json, pytest
from pathlib import Path
from fastapi.testclient import TestClient
from unittest import mock

IMPORT_PATH = "src.client.app.app"    #  ← đúng đường dẫn mới
with mock.patch("mlflow.tracking.MlflowClient") as FakeClient:
    FakeClient.return_value.get_latest_versions.return_value = [
        mock.Mock(version="1")
    ]
    FakeClient.return_value.download_artifacts.return_value = "/tmp"
    
@pytest.fixture(scope="session")
def client():
    os.environ.setdefault("MODEL_NAME",  "ci_XGB")
    os.environ.setdefault("MODEL_TYPE",  "xgb")
    os.environ.setdefault("MLFLOW_ENDPOINT", os.environ["MLFLOW_TRACKING_URI"])

    from importlib import import_module, reload
    mod = import_module(IMPORT_PATH)
    reload(mod)
    return TestClient(mod.app)

# ---------------------------------------------------------------------
def _payload():
    fp = Path(__file__).with_name("sample_payload.json")
    return json.loads(fp.read_text())

def _strip(t):      # remove timing
    t = dict(t)
    t.pop("inference_time_ms", None)
    return t

def test_prediction(client):
    res = client.post("/Prediction", json=_payload())
    assert res.status_code == 200
    data = _strip(res.json())
    assert data["predictions"][0]["result"] in ("Accept", "Decline")
    assert 0.0 <= data["metrics"]["avg_confidence"] <= 1.0

def test_prediction_by_id(client):
    res = client.post("/Prediction-by-id", params={"id": 100001})
    assert res.status_code == 200
    data = _strip(res.json())
    assert "predictions" in data
