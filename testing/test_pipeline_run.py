import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
from kfp.components import load_component_from_file
from kfp import dsl

# Add src/ to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

@patch("mlflow.xgboost.log_model")
@patch("mlflow.lightgbm.log_model")
@patch("mlflow.register_model")
@patch("mlflow.get_artifact_uri")
@patch("mlflow.log_artifacts")
@patch("mlflow.log_metric")
@patch("mlflow.log_params")
@patch("mlflow.start_run")
@patch("mlflow.set_experiment")
@patch("mlflow.set_tracking_uri")
@patch("minio.Minio")
def test_underwriting_pipeline_subprocess(
    mock_minio,
    mock_set_tracking_uri,
    mock_set_experiment,
    mock_start_run,
    mock_log_params,
    mock_log_metric,
    mock_log_artifacts,
    mock_get_artifact_uri,
    mock_register_model,
    mock_log_model_lgbm,
    mock_log_model_xgb,
    fake_raw_items,
    mock_minio_client,
    mock_mlflow_client,
):
    # MinIO
    mock_minio.return_value = mock_minio_client
    mock_start_run.return_value.__enter__.return_value = mock_mlflow_client

    ROOT_DIR = Path(__file__).resolve().parents[1] 
    components_dir = ROOT_DIR / "src" / "kfp_outside" / "components"

    dataloader_op = load_component_from_file(components_dir / "dataloader.yaml")
    preprocess_op = load_component_from_file(components_dir / "preprocess.yaml")
    modeling_op = load_component_from_file(components_dir / "model.yaml")

    @dsl.pipeline(name="UnderwritingWorkflowTest")
    def test_pipeline():
        raw_tr = dataloader_op(
            minio_endpoint="mock",
            minio_access_key="access",
            minio_secret_key="secret",
            bucket_name="bucket",
            object_name="raw/train.csv",
        )
        raw_te = dataloader_op(
            minio_endpoint="mock",
            minio_access_key="access",
            minio_secret_key="secret",
            bucket_name="bucket",
            object_name="raw/test.csv",
        )

        prep = preprocess_op(
            train_csv=raw_tr.outputs["output"],
            test_csv=raw_te.outputs["output"],
            minio_endpoint="mock",
            minio_access_key="access",
            minio_secret_key="secret",
            bucket_name="bucket",
            dest_train_object="processed/train.csv",
            dest_test_object="processed/test.csv",
            n_features_to_select="auto",
            data_version="v1",
        ).after(raw_te)

        proc_tr = dataloader_op(
            minio_endpoint="mock",
            minio_access_key="access",
            minio_secret_key="secret",
            bucket_name="bucket",
            object_name=prep.outputs["train_key"],
        ).after(prep)
        proc_te = dataloader_op(
            minio_endpoint="mock",
            minio_access_key="access",
            minio_secret_key="secret",
            bucket_name="bucket",
            object_name=prep.outputs["test_key"],
        ).after(prep)

        modeling_op(
            minio_endpoint="mock",
            minio_access_key="access",
            minio_secret_key="secret",
            train_csv=proc_tr.outputs["output"],
            test_csv=proc_te.outputs["output"],
            model_name="xgb",
            version="ci",
            experiment_name="CI_Exp",
        ).after(proc_te)

    from kfp import local
    from kfp.local import SubprocessRunner

    pipeline_output = tempfile.mkdtemp()
    local.init(runner=SubprocessRunner(), pipeline_root=pipeline_output)
    test_pipeline()

    mock_register_model.assert_called_once()
