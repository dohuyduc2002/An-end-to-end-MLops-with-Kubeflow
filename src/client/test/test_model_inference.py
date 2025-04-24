import pytest
from unittest.mock import MagicMock


@pytest.fixture(scope="module")
def mock_client_factory():
    def _create_mock_client(expected_output):
        mock = MagicMock()
        mock.post.return_value.status_code = 200
        mock.post.return_value.json.return_value = {
            "predictions": expected_output["predictions"],
            "metrics": expected_output["metrics"],
            "inference_time_ms": 42.0
        }
        return mock
    return _create_mock_client

def test_prediction_with_expected_result(mock_client_factory, sample_payload, expected_prediction_output):
    client = mock_client_factory(expected_prediction_output)
    response = client.post("/Prediction", json=sample_payload)
    assert response.status_code == 200
    result = response.json()
    result.pop("inference_time_ms", None)

    pred_actual = result["predictions"][0]
    pred_expected = expected_prediction_output["predictions"][0]
    for key in pred_expected:
        if isinstance(pred_expected[key], float):
            assert round(pred_actual[key], 4) == round(pred_expected[key], 4)
        else:
            assert pred_actual[key] == pred_expected[key]
