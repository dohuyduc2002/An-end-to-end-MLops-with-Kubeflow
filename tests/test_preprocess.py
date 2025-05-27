import pytest
import pandas as pd
import uuid
from pathlib import Path
from kfp.dsl import Dataset, Artifact
from tests.utils import make_test_artifact
from kfp_outside.script.preprocess import preprocess


@pytest.mark.unittest
def test_preprocess(tmp_path: Path, fake_csv: Path):
    tr_path = tmp_path / "train.csv"
    te_path = tmp_path / "test.csv"
    tr_path.write_text(fake_csv.read_text())
    te_path.write_text(fake_csv.read_text())

    kfp_dataset, kfp_artifact = make_test_artifact(Dataset), make_test_artifact(Artifact)
    train_csv = kfp_dataset(uri=str(tr_path))
    test_csv = kfp_dataset(uri=str(te_path))
    output_train_csv = kfp_dataset(uri=str(tmp_path / "out_train.csv"))
    output_test_csv = kfp_dataset(uri=str(tmp_path / "out_test.csv"))
    transformer_joblib = kfp_artifact(uri=str(tmp_path / "transformer.joblib"))
    mlflow_run_id = kfp_artifact(uri=str(tmp_path / "mlflow_run_id.txt"))

    keys = preprocess.python_func(
        train_csv=train_csv,
        test_csv=test_csv,
        output_train_csv=output_train_csv,
        output_test_csv=output_test_csv,
        transformer_joblib=transformer_joblib,
        mlflow_run_id=mlflow_run_id,
        minio_endpoint="fake:9000",
        minio_access_key="a",
        minio_secret_key="b",
        mlflow_endpoint="fake:5000",
        parent_run_name="unittest_parent_run_name",
        dest_train_object="train.csv",
        dest_test_object="test.csv",
        n_features_to_select="auto",
        data_version="unittest",
        experiment_name="unittest_experiment",
    )

    assert Path(transformer_joblib.path).exists()
    assert Path(output_train_csv.path).exists()
    assert Path(output_test_csv.path).exists()

    df_tr = pd.read_csv(output_train_csv.path)
    df_te = pd.read_csv(output_test_csv.path)

    assert "TARGET" in df_tr.columns
    assert df_tr.shape[0] > 0
    assert df_te.shape[0] > 0

    run_id = Path(mlflow_run_id.path).read_text().strip()

    try:
        uuid_obj = uuid.UUID(run_id)
        assert str(uuid_obj) == run_id
    except ValueError:
        assert False, f"Invalid run_id format: {run_id}"
