from unittest.mock import MagicMock


def build_mock_mlflow():
    mock_mlflow = MagicMock(name="mlflow")

    # Tracking API
    mock_mlflow.set_tracking_uri = MagicMock()
    mock_mlflow.set_experiment = MagicMock()
    mock_mlflow.end_run = MagicMock()
    mock_mlflow.log_params = MagicMock()
    mock_mlflow.log_metric = MagicMock()
    mock_mlflow.log_artifact = MagicMock()
    mock_mlflow.log_artifacts = MagicMock()
    mock_mlflow.register_model = MagicMock()
    mock_mlflow.tracking = MagicMock()

    # Context manager for start_run
    def _start_run(*args, **kwargs):
        run_mock = MagicMock()
        run_mock.info.run_id = "run_id_mock"
        cm = MagicMock()
        cm.__enter__.return_value = run_mock
        cm.__exit__.return_value = False
        return cm

    mock_mlflow.start_run.side_effect = _start_run

    mock_mlflow.xgboost = MagicMock()
    mock_mlflow.xgboost.log_model = MagicMock()
    mock_mlflow.lightgbm = MagicMock()
    mock_mlflow.lightgbm.log_model = MagicMock()

    mock_mlflow.tracking = MagicMock()
    mock_mlflow.tracking.MlflowClient = MagicMock()
    return mock_mlflow
