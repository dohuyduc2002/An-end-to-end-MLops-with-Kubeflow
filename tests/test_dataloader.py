import pytest
from pathlib import Path
from kfp.dsl import Dataset
from kfp_outside.script.dataloader import dataloader
from tests.utils import make_test_artifact


@pytest.mark.unittest
def test_dataloader(tmp_path: Path):
    kfp_dataset = make_test_artifact(Dataset)
    artifact = kfp_dataset(uri=str(tmp_path / "file.csv"))

    dataloader.python_func(
        minio_endpoint="fake:9000",
        minio_access_key="id",
        minio_secret_key="key",
        bucket_name="anything",
        object_name="obj.csv",
        output=artifact,
    )

    assert Path(artifact.path).exists()
