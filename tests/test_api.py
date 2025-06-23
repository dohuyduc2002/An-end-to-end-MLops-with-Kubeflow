import pytest
from fastapi.testclient import TestClient

@pytest.fixture
def client(patch_env_api):
    from deprecated.main import app
    return TestClient(app)

@pytest.mark.api
def test_health(client):
    res = client.get("/")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


@pytest.mark.api
def test_prediction_success(
    client, sample_payload, sample_result, strip_time_field, compare_schema
):
    res = client.post("/Prediction", json=sample_payload)
    assert res.status_code == 200
    body = res.json()
    assert isinstance(body["predictions"], list)
    body = strip_time_field(body)
    compare_schema(body, sample_result)


@pytest.mark.api
def test_prediction_by_id(client, sample_result, strip_time_field, compare_schema):
    res = client.post("/Prediction-by-id", params={"id": 100001})
    assert res.status_code == 200
    body = res.json()
    body = strip_time_field(body)
    compare_schema(body, sample_result)


@pytest.mark.api
def test_prediction_by_id_not_found(client):
    res = client.post("/Prediction-by-id", params={"id": 999999})
    assert res.json() == {"error": "ID 999999 not found"}


@pytest.mark.api
def test_prediction_by_id_invalid_id(client):
    res = client.post("/Prediction-by-id", params={"id": "invalid"})
    assert res.status_code == 422
    body = res.json()
    assert body["detail"][0]["loc"] == ["query", "id"]
    assert "integer" in body["detail"][0]["type"] or "int" in body["detail"][0]["type"]
