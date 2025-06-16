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

    # Context manager for start_run
    def _start_run(*args, **kwargs):
        run_mock = MagicMock()
        run_mock.info.run_id = "run_id_mock"
        cm = MagicMock()
        cm.__enter__.return_value = run_mock
        cm.__exit__.return_value = False
        return cm

    mock_mlflow.start_run.side_effect = _start_run

    # Register model trả về object
    reg = MagicMock()
    reg.name, reg.version = "mock_model", 1
    mock_mlflow.register_model.return_value = reg

    return mock_mlflow


def build_mock_optuna():
    fake_trial = MagicMock()
    fake_trial.number = 0
    fake_trial.user_attrs = {"mlflow_run_id": "run_1"}
    fake_trial.set_user_attr = MagicMock()

    fake_study = MagicMock()
    fake_study.best_trial = fake_trial
    fake_study.optimize = MagicMock(side_effect=lambda objective, n_trials: None) # patch objective call to do nothing

    return fake_study
