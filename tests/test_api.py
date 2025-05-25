import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(patch_minio_and_mlflow):
    from client.api.main import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


@pytest.mark.unittest
def test_health(client):
    res = client.get("/")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


@pytest.mark.unittest
def test_prediction_success(client, sample_payload, expected_result, strip_time_field, compare_results):
    res = client.post("/Prediction", json=sample_payload)
    body = res.json()

    assert res.status_code == 200
    assert isinstance(body["predictions"], list)
    assert len(body["predictions"]) == len(sample_payload)

    body = strip_time_field(res.json())
    compare_results(body, expected_result, decimal=3)


@pytest.mark.unittest
def test_prediction_by_id(client, expected_result, strip_time_field, compare_results):
    """
    Uses the row with SK_ID_CURR == 100001 embedded in the FakeMinio fixture.
    """
    res = client.post("/Prediction-by-id", params={"id": 100001})
    body = res.json()

    body = strip_time_field(res.json())
    compare_results(body, expected_result, decimal=3)


@pytest.mark.unittest
def test_metrics_handler_update():
    from client.api.main import MetricsHandler

    metrics_handler = MetricsHandler()
    assert metrics_handler._avg_entropy == 0.0
    assert metrics_handler._avg_confidence == 0.0

    metrics_handler.update([0.1, 0.2, 0.3], [0.7, 0.8, 0.9])
    assert metrics_handler._avg_entropy == pytest.approx(0.2)
    assert metrics_handler._avg_confidence == pytest.approx(0.8)
