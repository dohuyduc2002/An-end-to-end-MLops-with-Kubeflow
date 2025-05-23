import json
import pytest
from pathlib import Path
from fastapi.testclient import TestClient


@pytest.fixture
def client(patch_api_joblib):
    import client.api.main as main

    return TestClient(app=main.app)


def _strip(t):
    t = dict(t)
    t.pop("inference_time_ms", None)
    return t


EXPECTED_RESPONSE = {
    "predictions": [
        {
            "result": "Accept",
            "prob_accept": 0.9988225102424622,
            "prob_decline": 0.0011774982558563352,
            "entropy": 0.0132,
            "confidence": 0.9988,
        }
    ],
    "metrics": {
        "avg_entropy": 0.013154886662960052,
        "avg_confidence": 0.9988225102424622,
    },
}


# === TEST: /Prediction ===
@pytest.mark.unittest
def test_prediction_success(client, patch_minio_and_mlflow, sample_payload):
    res = client.post("/Prediction", json=sample_payload)
    actual = _strip(res.json())

    if actual != EXPECTED_RESPONSE:
        print("❌ Prediction mismatch!")
        print("📌 Actual:", json.dumps(actual, indent=2))
        print("✅ Expected:", json.dumps(EXPECTED_RESPONSE, indent=2))

    assert res.status_code == 200
    assert actual == EXPECTED_RESPONSE


@pytest.mark.unittest
def test_prediction_missing_column(
    client, patch_minio_and_mlflow, sample_payload, exclude_opentelemetry_imports
):
    for row in sample_payload:
        row.pop("EXT_SOURCE_1", None)

    res = client.post("/Prediction", json=sample_payload)

    print("🔍 Response with missing EXT_SOURCE_1:")
    print(json.dumps(res.json(), indent=2))

    assert _strip(res.json()) == EXPECTED_RESPONSE


# === TEST: /Prediction-by-id ===
@pytest.mark.unittest
def test_prediction_by_id_success(
    client, patch_minio_and_mlflow, patch_api_joblib, exclude_opentelemetry_imports
):
    res = client.post("/Prediction-by-id", params={"id": 100001})
    assert res.status_code == 200
    assert _strip(res.json()) == EXPECTED_RESPONSE


@pytest.mark.unittest
def test_prediction_by_id_not_found(
    client, patch_minio_and_mlflow, patch_api_joblib, exclude_opentelemetry_imports
):
    res = client.post("/Prediction-by-id", params={"id": 999999999})
    assert res.status_code in (404, 200)
    assert "not found" in res.text.lower()


# === TEST: / (health) ===
@pytest.mark.unittest
def test_healthcheck(client):
    res = client.get("/")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}
