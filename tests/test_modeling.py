import json
from pathlib import Path
import pandas as pd
import pytest
from kfp.dsl import Artifact, Dataset
from kubeflow_nb.pipeline.scripts.modeling import modeling

from tests.test_utils import make_test_artifact


@pytest.mark.kfp_components
def test_modeling_component(tmp_path: Path, fake_csv: Path, patch_env_kfp):
    df = pd.read_csv(fake_csv).select_dtypes(include=["number", "bool"])

    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    df.to_csv(train_path, index=False)
    df.to_csv(test_path, index=False)

    parent_run_path = tmp_path / "parent_run.txt"
    parent_run_path.write_text("run_0")  
    kfp_dataset, kfp_artifact = make_test_artifact(Dataset), make_test_artifact(
        Artifact
    )
    train_csv = kfp_dataset(uri=str(train_path))
    test_csv = kfp_dataset(uri=str(test_path))
    model_joblib = kfp_artifact(uri=str(tmp_path / "model.joblib"))
    registered_model = kfp_artifact(uri=str(tmp_path / "registered.json"))
    mlflow_run_id = kfp_artifact(uri=str(parent_run_path))

    # Gọi component
    modeling.python_func(
        train_csv=train_csv,
        test_csv=test_csv,
        model_joblib=model_joblib,
        registered_model=registered_model,
        minio_endpoint="fake:9000",
        minio_access_key="abc",
        minio_secret_key="123",
        mlflow_run_id=mlflow_run_id,
        mlflow_endpoint="fake:5000",
        experiment_name="unit_exp",
        model_name="xgb",  # hoặc "lgbm"
        suffix="unittest",
    )

    reg_path = Path(registered_model.path)
    assert reg_path.exists(), "registered_model artifact not created"
    data = json.loads(reg_path.read_text())

    assert data["parent_run"] == "run_0"
    assert data["registered"]["name"]
