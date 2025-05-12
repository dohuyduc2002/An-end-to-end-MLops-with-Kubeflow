import os
import mlflow
from tempfile import gettempdir
from kfp import local
from src.kfp_outside.pipeline import underwriting_pipeline


def test_full_pipeline(sample_data, mlflow_client):
    os.environ.pop("DOCKER_HOST", None)

    pipeline_output_dir = os.path.join(gettempdir(), "kfp_outputs")
    os.makedirs(pipeline_output_dir, exist_ok=True)

    local.init(
        runner=local.DockerRunner(),
        pipeline_root=pipeline_output_dir
    )

    bucket, tr_key, te_key = sample_data

    underwriting_pipeline(
        minio_endpoint=os.environ["MINIO_ENDPOINT"],
        minio_access_key=os.environ["MINIO_ACCESS_KEY"],
        minio_secret_key=os.environ["MINIO_SECRET_KEY"],
        bucket_name=bucket,
        raw_train_object=tr_key,
        raw_test_object=te_key,
        dest_train_object="processed/train.csv",
        dest_test_object="processed/test.csv",
        model_name="xgb",
        version="ci",
        n_features_to_select="auto",
        data_version="v1",
        experiment_name="CI_Exp",
    )

    registered_models = [m.name for m in mlflow_client.list_registered_models()]
    assert "ci_XGB" in registered_models
