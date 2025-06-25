import pytest
from fastapi.testclient import TestClient
@pytest.fixture
def client(patch_env):
    from src.client.api.main import app
    return TestClient(app)


@pytest.mark.api
def test_health(client):
    res = client.get("/")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


@pytest.mark.api
def test_prediction_success(
    client, sample_payload):
    res = client.post("/prediction", json=sample_payload)
    assert res.status_code == 200


@pytest.mark.api
def test_prediction_invalid_schema(client):
    payload = [{"invalid_field": 123}]
    res = client.post("/prediction", json=payload)
    assert res.status_code == 422  


@pytest.mark.api
def test_prediction_by_id_valid(client,patch_minio):
    resp = client.post("/prediction-by-id", params={"id": 100001})
    assert resp.status_code == 200


@pytest.mark.api
def test_prediction_by_id_not_found(client, patch_minio):
    res = client.post("/prediction-by-id", params={"id": 999999})
    assert res.status_code == 404  

@pytest.mark.api
def test_data_monitor_success(mocker,client):
    mock_ws = mocker.patch("src.client.api.main.RemoteWorkspace")
    mock_map_data = mocker.patch("src.client.api.main.map_evidently_data")
    mock_custom_report = mocker.patch("src.client.api.main.custom_evidently_report")
    
    mock_ws.return_value.search_project.return_value = []
    mock_ws.return_value.create_project.return_value.id = "proj_id"
    mock_map_data.return_value = ([], [])
    mock_custom_report.return_value = {}
    mock_ws.return_value.add_run.return_value = None

    res = client.get("/data-monitor")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "stored"
    assert body["project_id"] == "proj_id"
    assert body["project_name"] == "credit_underwriting"
