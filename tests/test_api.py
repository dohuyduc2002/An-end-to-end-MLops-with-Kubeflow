import pytest
from fastapi import HTTPException


@pytest.fixture
def client(fake_lifespan, mock_predict_batch, mock_predict_by_id, mock_minio_client):
    from src.client.api.main import create_app
    from fastapi.testclient import TestClient

    with TestClient(create_app(lifespan=fake_lifespan)) as client:
        yield client


@pytest.mark.api
def test_health(client):
    res = client.get("/")

    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


@pytest.mark.api
def test_prediction_success(client, sample_payload):
    res = client.post("/prediction", json=sample_payload)
    preds = res.json()["predictions"]

    assert res.status_code == 200
    assert preds == [
        {"result": "Accept", "prob_accept": 0.7, "prob_decline": 0.3},
        {"result": "Decline", "prob_accept": 0.2, "prob_decline": 0.8},
    ]


@pytest.mark.api
def test_prediction_invalid_schema(client):
    payload = [{"invalid_field": 123}]
    res = client.post("/prediction", json=payload)

    assert res.status_code == 422


@pytest.mark.api
def test_prediction_by_id_valid(client):
    resp = client.post("/prediction-by-id", params={"id": 100001})

    assert resp.status_code == 200
    assert resp.json()["predictions"] == [
        {"result": "Accept", "prob_accept": 0.95, "prob_decline": 0.05}
    ]


@pytest.mark.api
def test_prediction_by_id_not_found(client, mocker):
    mocker.patch(
        "src.client.api.main.predict_by_id",
        side_effect=HTTPException(status_code=404, detail="ID not found"),
    )
    res = client.post("/prediction-by-id", params={"id": 999999})

    assert res.status_code == 404
