import pytest
import pandas as pd
import json
from pathlib import Path
from kfp.dsl import Dataset, Artifact
from tests.utils import make_test_artifact

from kfp_outside.script.modeling import modeling


def filter_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include=["number", "bool"])


@pytest.mark.unittest
def test_modeling(tmp_path: Path, fake_csv: Path):
    # Read and filter the input CSV
    df = pd.read_csv(fake_csv)
    df = filter_numeric_columns(df)

    # Create train/test CSV files
    tr_path = tmp_path / "train.csv"
    te_path = tmp_path / "test.csv"
    df.to_csv(tr_path, index=False)
    df.to_csv(te_path, index=False)

    # Write a dummy MLflow run ID
    mlflow_run_id_path = tmp_path / "mlflow_run_id.txt"
    mlflow_run_id_path.write_text("dummy_run_id")

    # Create KFP-compatible artifacts
    kfp_dataset, kfp_artifact = make_test_artifact(Dataset), make_test_artifact(
        Artifact
    )
    train_csv = kfp_dataset(uri=str(tr_path))
    test_csv = kfp_dataset(uri=str(te_path))
    model_joblib = kfp_artifact(uri=str(tmp_path / "mdl.joblib"))
    registered_model = kfp_artifact(uri=str(tmp_path / "name.txt"))
    mlflow_run_id = kfp_artifact(uri=str(mlflow_run_id_path))

    # Run the modeling function
    modeling.python_func(
        train_csv=train_csv,
        test_csv=test_csv,
        model_joblib=model_joblib,
        registered_model=registered_model,
        mlflow_run_id=mlflow_run_id,
        minio_endpoint="fake:9000",
        minio_access_key="id",
        minio_secret_key="key",
        mlflow_endpoint="fake:5000",
        experiment_name="unittest_experiment",
        model_name="xgb", # xgb or lgbm only
        suffix="unittest",
    )

    reg_path = Path(registered_model.path)
    assert reg_path.exists()

    data = json.loads(reg_path.read_text())
    assert {"parent_run", "best_trial", "best_trial_run", "registered"} <= data.keys()

    assert data["registered"]["name"].endswith("_unittest")
