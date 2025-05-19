import pytest
import pandas as pd
from pathlib import Path
from kfp.dsl import Dataset, Model
from testing.utils import make_test_artifact
from kfp_outside.script.preprocess import preprocess

@pytest.mark.unit
def test_preprocess(tmp_path: Path, fake_csv: Path):
    tr_path = tmp_path / "train.csv"
    te_path = tmp_path / "test.csv"
    tr_path.write_text(fake_csv.read_text())
    te_path.write_text(fake_csv.read_text())

    DS, MD = make_test_artifact(Dataset), make_test_artifact(Model)
    train_csv = DS(uri=str(tr_path))
    test_csv = DS(uri=str(te_path))
    output_train_csv = DS(uri=str(tmp_path / "out_train.csv"))
    output_test_csv  = DS(uri=str(tmp_path / "out_test.csv"))
    transformer_joblib = MD(uri=str(tmp_path / "transformer.joblib"))

    keys = preprocess.python_func(
        train_csv=train_csv,
        test_csv=test_csv,
        output_train_csv=output_train_csv,
        output_test_csv=output_test_csv,
        transformer_joblib=transformer_joblib,
        minio_endpoint="fake:9000",
        minio_access_key="a",
        minio_secret_key="b",
        bucket_name="bk",
        dest_train_object="train.csv",
        dest_test_object="test.csv",
    )

    assert Path(transformer_joblib.path).exists()
    assert Path(output_train_csv.path).exists()
    assert Path(output_test_csv.path).exists()

    df_tr = pd.read_csv(output_train_csv.path)
    df_te = pd.read_csv(output_test_csv.path)

    assert "TARGET" in df_tr.columns
    assert df_tr.shape[0] > 0
    assert df_te.shape[0] > 0
    assert keys[0].startswith("train_")
    assert keys[1].startswith("test_")
