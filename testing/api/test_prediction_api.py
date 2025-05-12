import os, json, pytest
from pathlib import Path
from fastapi.testclient import TestClient
from fastapi import HTTPException

IMPORT_PATH = "src.client.app.app"

@pytest.fixture(scope="session")
def client():
    os.environ.setdefault("MODEL_NAME", "ci_XGB")
    os.environ.setdefault("MODEL_TYPE", "xgb")
    os.environ.setdefault("MLFLOW_ENDPOINT", os.environ["MLFLOW_TRACKING_URI"])

    from importlib import import_module, reload
    mod = import_module(IMPORT_PATH)
    reload(mod)
    return TestClient(mod.app)

def _payload():
    fp = Path(__file__).with_name("sample_payload.json")
    return json.loads(fp.read_text())

def _strip(t):
    t = dict(t)
    t.pop("inference_time_ms", None)
    return t

EXPECTED_RESPONSE = {
    "predictions": [
        {
            "result": "Accept",
            "prob_accept": 0.9988268613815308,
            "prob_decline": 0.0011731224367395043,
            "entropy": 0.0131,
            "confidence": 0.9988
        }
    ],
    "metrics": {
        "avg_entropy": 0.013112341053783894,
        "avg_confidence": 0.9988268613815308
    }
}

def test_prediction_success(client, monkeypatch):
    monkeypatch.setattr("src.client.app.app.infer", lambda x: EXPECTED_RESPONSE)
    res = client.post("/Prediction", json=_payload())
    assert res.status_code == 200
    assert _strip(res.json()) == EXPECTED_RESPONSE

def test_prediction_missing_column(client):
    payload = _payload()
    for row in payload["data"]:
        row.pop("EXT_SOURCE_1", None)  

    res = client.post("/Prediction", json=payload)
    assert res.status_code in (400, 422)
    assert "missing" in res.text.lower() or "column" in res.text.lower()

def test_prediction_by_id_success(client, monkeypatch):
    monkeypatch.setattr("src.client.app.app.infer_by_id", lambda id: EXPECTED_RESPONSE)
    res = client.post("/Prediction-by-id", params={"id": 100001})
    assert res.status_code == 200
    assert _strip(res.json()) == EXPECTED_RESPONSE

def test_prediction_by_id_not_found(client, monkeypatch):
    def raise_not_found(id):
        raise HTTPException(status_code=404, detail=f"ID {id} not found")
    monkeypatch.setattr("src.client.app.app.infer_by_id", raise_not_found)

    res = client.post("/Prediction-by-id", params={"id": 99999})
    assert res.status_code == 404
    assert "not found" in res.text.lower()
