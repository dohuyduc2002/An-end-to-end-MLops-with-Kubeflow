from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Artifact
from component_utils import BASE_IMAGE, TARGET_IMAGE


@dsl.component(base_image=BASE_IMAGE, target_image=TARGET_IMAGE)
def feat_selector(
    df_train_binned_csv: Input[Dataset],
    df_test_binned_csv: Input[Dataset],
    base_mlflow_run_id: Input[Artifact],
    transformer_joblib: Output[Artifact],
    output_train_csv: Output[Dataset],
    output_test_csv: Output[Dataset],
    mlflow_run_id: Output[Artifact],
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    mlflow_endpoint: str,
    n_features_to_select: str,
    experiment_name: str,
):

    import os
    import pandas as pd
    import joblib
    from pathlib import Path
    import mlflow

    from component_utils import fit_selector

    os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://{minio_endpoint}"
    os.environ["AWS_ACCESS_KEY_ID"] = minio_access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = minio_secret_key
    os.environ["MLFLOW_ENDPOINT"] = f"http://{mlflow_endpoint}"

    # ========== Pipeline ==========
    df_train_binned = pd.read_csv(df_train_binned_csv.path)
    df_test_binned = pd.read_csv(df_test_binned_csv.path)

    y = df_train_binned["TARGET"]
    df_train_binned_no_target = df_train_binned.drop(columns=["TARGET"])

    selector, out_train, out_test = fit_selector(
        df_train_binned_no_target, df_test_binned, y, n_features_to_select
    )

    # Save and log artifacts to MLflow
    mlflow.set_tracking_uri(os.environ["MLFLOW_ENDPOINT"])
    mlflow.set_experiment(experiment_name)

    run_id = Path(base_mlflow_run_id.path).read_text().strip()

    with mlflow.start_run(run_id=run_id) as parent:
        parent_id = parent.info.run_id  # Save the parent run ID for later use
        joblib.dump(selector, "/tmp/feat_selector.joblib")
        mlflow.log_artifact("/tmp/feat_selector.joblib", artifact_path="prep")

    # Save to KFP artifact
    joblib.dump(
        {"selector": selector},
        transformer_joblib.path,
    )
    out_train.to_csv(output_train_csv.path, index=False)
    out_test.to_csv(output_test_csv.path, index=False)
    Path(mlflow_run_id.path).parent.mkdir(parents=True, exist_ok=True)
    Path(mlflow_run_id.path).write_text(parent_id)
