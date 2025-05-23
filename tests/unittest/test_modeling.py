import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from kfp.dsl import Dataset, Artifact
from tests.utils import make_test_artifact

from kfp_outside.script.modeling import modeling


def filter_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include=["number", "bool"])


@pytest.mark.unittest
def test_modeling(tmp_path: Path, fake_csv: Path):
    df = pd.read_csv(fake_csv)
    df = filter_numeric_columns(df)

    tr_path = tmp_path / "train.csv"
    te_path = tmp_path / "test.csv"
    df.to_csv(tr_path, index=False)
    df.to_csv(te_path, index=False)

    kfp_dataset, kfp_artifact = make_test_artifact(Dataset), make_test_artifact(Artifact)
    train_csv = kfp_dataset(uri=str(tr_path))
    test_csv = kfp_dataset(uri=str(te_path))
    out_model = kfp_artifact(uri=str(tmp_path / "mdl.joblib"))
    reg_model = str(tmp_path / "name.txt")

    modeling.python_func(
        train_csv=train_csv,
        test_csv=test_csv,
        model_joblib=out_model,
        registered_model=reg_model,
        minio_endpoint="fake:9000",
        minio_access_key="id",
        minio_secret_key="key",
        mlflow_endpoint="http://fake:5000",
        version="unittest",
    )

    assert Path(out_model.path).exists()
    assert Path(reg_model).exists()
    assert Path(reg_model).read_text().startswith("unittest_")
