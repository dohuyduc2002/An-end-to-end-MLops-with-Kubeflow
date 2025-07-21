from kfp import dsl
from kfp.components import load_component_from_file
from pathlib import Path

COMP_DIR = (Path(__file__).parent / "scripts" / "component_metadata").resolve()

dataloader_op = load_component_from_file(COMP_DIR / "dataloader.yaml")
binning_op = load_component_from_file(COMP_DIR / "binning.yaml")
selector_op = load_component_from_file(COMP_DIR / "feat_selector.yaml")
modeling_op = load_component_from_file(COMP_DIR / "modeling.yaml")
notify_slack_op = load_component_from_file(COMP_DIR / "slack_notification.yaml")


@dsl.pipeline(
    name="UnderwritingWorkflow",
    description="Download raw → preprocess → download processed → train & register",
)
def underwriting_pipeline(
    job_name: str,
    slack_channel: str,
    slack_bot_token: str,
    minio_endpoint: str,
    mlflow_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    bucket_name: str,
    raw_train_object: str,
    raw_test_object: str,
    parent_run_name: str,
    model_type: str,
    suffix: str,
    experiment_name: str,
    n_features_to_select: str,
    iv_min: float,
    iv_max: float,
    missing_thres: float,
):
    # 1 Download raw train
    raw_tr = dataloader_op(
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        bucket_name=bucket_name,
        object_name=raw_train_object,
    ).set_caching_options(enable_caching=False)

    # 2 Download raw test
    raw_te = dataloader_op(
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        bucket_name=bucket_name,
        object_name=raw_test_object,
    ).set_caching_options(enable_caching=False)

    # 3 binning
    binner = (
        binning_op(
            train_csv=raw_tr.outputs["output"],
            test_csv=raw_te.outputs["output"],
            iv_min=iv_min,
            iv_max=iv_max,
            missing_thres=missing_thres,
            minio_endpoint=minio_endpoint,
            minio_access_key=minio_access_key,
            minio_secret_key=minio_secret_key,
            mlflow_endpoint=mlflow_endpoint,
            parent_run_name=parent_run_name,
            experiment_name=experiment_name,
        )
        .after(raw_tr, raw_te)
        .set_caching_options(enable_caching=False)
    )

    # 4 Selector
    selector = (
        selector_op(
            df_train_binned_csv=binner.outputs["df_train_binned_csv"],
            df_test_binned_csv=binner.outputs["df_test_binned_csv"],
            base_mlflow_run_id=binner.outputs["base_mlflow_run_id"],
            n_features_to_select=n_features_to_select,
            mlflow_endpoint=mlflow_endpoint,
            minio_endpoint=minio_endpoint,
            minio_access_key=minio_access_key,
            minio_secret_key=minio_secret_key,
            experiment_name=experiment_name,
        )
        .after(binner)
        .set_caching_options(enable_caching=False)
    )

    # 4 Modeling
    modeling_op(
        mlflow_endpoint=mlflow_endpoint,
        processed_train_csv=selector.outputs["output_train_csv"],
        processed_test_csv=selector.outputs["output_test_csv"],
        mlflow_run_id=selector.outputs["mlflow_run_id"],
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        model_type=model_type,
        suffix=suffix,
        experiment_name=experiment_name,
    ).after(selector).set_caching_options(enable_caching=False)

    # 5 Notify Slack
    message = f"Pipeline completed successfully with job: {job_name} for {model_type}"

    notify_slack_op(
        slack_channel=slack_channel, message=message, slack_bot_token=slack_bot_token
    ).after(modeling_op).set_caching_options(enable_caching=False)
