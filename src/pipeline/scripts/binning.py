from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Artifact
from pathlib import Path
from component_utils import BASE_IMAGE, TARGET_IMAGE


@dsl.component(base_image=BASE_IMAGE, target_image=TARGET_IMAGE)
def binning(
    train_csv: Input[Dataset],
    test_csv: Input[Dataset],
    transformer_joblib: Output[Artifact],
    df_train_binned_csv: Output[Dataset],
    df_test_binned_csv: Output[Dataset],
    base_mlflow_run_id: Output[Artifact],
    iv_min: float,
    iv_max: float,
    missing_thres: float,
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    mlflow_endpoint: str,
    parent_run_name: str,
    experiment_name: str,
):

    import os
    import pandas as pd
    import joblib
    from pathlib import Path
    import mlflow
    from datetime import datetime
    
    from component_utils import get_lists, iv_score, select_survivors, fit_binning

    os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://{minio_endpoint}"
    os.environ["AWS_ACCESS_KEY_ID"] = minio_access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = minio_secret_key
    os.environ["MLFLOW_ENDPOINT"] = f"http://{mlflow_endpoint}"

    # ========== Pipeline ==========
    df_train = pd.read_csv(train_csv.path)
    df_test = pd.read_csv(test_csv.path)

    cat_cols, num_cols = get_lists(df_train)
    survivors = select_survivors(df_train, cat_cols, num_cols, iv_min, iv_max, missing_thres)
    y = df_train["TARGET"]
    X_train, X_test = df_train.drop("TARGET", axis=1), df_test.copy()

    opt_binning_process, df_train_binned, df_test_binned = fit_binning(
        X_train, X_test, y, survivors, cat_cols
    )

    mlflow.set_tracking_uri(os.environ["MLFLOW_ENDPOINT"])
    mlflow_client = mlflow.MlflowClient()

    # Get or create experiment
    experiment = mlflow_client.get_experiment_by_name(experiment_name)
    if experiment is not None:
        experiment_id = experiment.experiment_id
    else:
        experiment_id = mlflow_client.create_experiment(experiment_name)

    now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    unique_run_name = f"{parent_run_name}-{now_str}"

    run = mlflow_client.create_run(experiment_id=experiment_id, run_name=unique_run_name)
    
    parent_id = run.info.run_id

    # Save artifacts
    joblib.dump(
        {"opt_binning_process": opt_binning_process},
        "/tmp/opt_binning_process.joblib",
    )

    mlflow_client.log_artifact(
        parent_id, "/tmp/opt_binning_process.joblib", artifact_path="prep"
    )

    # Save to KFP artifact
    Path(base_mlflow_run_id.path).write_text(parent_id)

    joblib.dump(
        {"opt_binning_process": opt_binning_process},
        transformer_joblib.path,
    )
    df_train_binned.to_csv(df_train_binned_csv.path, index=False)
    df_test_binned.to_csv(df_test_binned_csv.path, index=False)
