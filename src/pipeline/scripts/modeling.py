from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Artifact
from component_utils import BASE_IMAGE, TARGET_IMAGE


@dsl.component(base_image=BASE_IMAGE, target_image=TARGET_IMAGE)
def modeling(
    processed_train_csv: Input[Dataset],
    processed_test_csv: Input[Dataset],
    model_joblib: Output[Artifact],
    registered_model: Output[Artifact],
    mlflow_run_id: Input[Artifact],
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    mlflow_endpoint: str,
    experiment_name: str,
    model_type: str,
    suffix: str,
):
    import os
    import json
    import mlflow
    import pandas as pd
    from pathlib import Path

    from component_utils import build_objective, run_optuna_study

    os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://{minio_endpoint}"
    os.environ["AWS_ACCESS_KEY_ID"] = minio_access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = minio_secret_key
    os.environ["MLFLOW_ENDPOINT"] = f"http://{mlflow_endpoint}"

    # ===== Pipeline =====
    mlflow.set_tracking_uri(f"http://{mlflow_endpoint}")
    mlflow.set_experiment(experiment_name)
    parent_id: str = Path(mlflow_run_id.path).read_text().strip()
    mlflow.end_run()  # ensure any existing run is closed

    df = pd.read_csv(processed_train_csv.path)
    X, y = df.drop("TARGET", axis=1), df["TARGET"]

    with mlflow.start_run(run_id=parent_id):
        objective = build_objective(X, y, model_type, suffix)
        best_trial, best_run_id = run_optuna_study(objective, n_trials=5)
        best_model_uri = f"runs:/{best_run_id}/model"
        registry = mlflow.register_model(best_model_uri, name=f"{model_type}_{suffix}")

    Path(registered_model.path).parent.mkdir(parents=True, exist_ok=True)
    Path(registered_model.path).write_text(
        json.dumps(
            {
                "parent_run": parent_id,
                "best_trial": best_trial.number,
                "best_trial_run": best_run_id,
                "registered": {"name": registry.name, "version": registry.version},
            },
            indent=2,
        )
    )
