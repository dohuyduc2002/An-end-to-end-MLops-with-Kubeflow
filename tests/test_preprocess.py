import pytest
from kfp.dsl import Dataset, Artifact
from tests.test_utils import make_test_artifact

from kubeflow_nb.pipeline.scripts.preprocess import preprocess


@pytest.mark.kfp_components
def test_preprocess(tmp_path, fake_csv, patch_env_kfp):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train_path.write_text(fake_csv.read_text())
    test_path.write_text(fake_csv.read_text())

    kfp_dataset, kfp_artifact = make_test_artifact(Dataset), make_test_artifact(
        Artifact
    )
    train_csv = kfp_dataset(uri=str(train_path)) #input artifact
    test_csv = kfp_dataset(uri=str(test_path))  # input artifact

    output_train_csv = kfp_dataset(uri=str(tmp_path / "out_train.csv")) # output artifact
    output_test_csv = kfp_dataset(uri=str(tmp_path / "out_test.csv"))  # output artifact
    transformer_joblib = kfp_artifact(
        uri=str(tmp_path / "transformer.joblib")
    )  # output artifact
    mlflow_run_id = kfp_artifact(
        uri=str(tmp_path / "mlflow_run_id.txt")
    )  # output artifact

    parent_run_name = "unittest_parent_run_name"

    keys = preprocess.python_func(
        train_csv=train_csv,
        test_csv=test_csv,
        output_train_csv=output_train_csv,
        output_test_csv=output_test_csv,
        minio_endpoint="fake:9000",
        minio_access_key="abc",
        minio_secret_key="123",
        transformer_joblib=transformer_joblib,
        mlflow_run_id=mlflow_run_id,
        mlflow_endpoint="abc:5000",
        parent_run_name=parent_run_name,
        n_features_to_select="auto",
        experiment_name="unittest_experiment",
    )

    with open(tmp_path / "mlflow_run_id.txt") as f: # assert mlflow_run_id artifact
        content = f.read().strip()
    assert content 
