import pytest
from pathlib import Path
from kfp.dsl import Dataset
from unittest.mock import patch, MagicMock
from tests.test_utils import make_test_artifact
from kubeflow_nb.pipeline.scripts.dataloader import dataloader


@pytest.mark.kfp_components
def test_dataloader(tmp_path: Path, patch_env_kfp):
    kfp_dataset = make_test_artifact(Dataset)
    artifact = kfp_dataset(uri=str(tmp_path / "file.csv"))

    with patch("minio.Minio") as MockMinio:
        mock_client = MagicMock()
        MockMinio.return_value = mock_client
        mock_client.fget_object.return_value = None

        dataloader.python_func(
            minio_endpoint="fake:9000",
            minio_access_key="id",
            minio_secret_key="key",
            bucket_name="anything",
            object_name="obj.csv",
            output=artifact,
        )

        mock_client.fget_object.assert_called_once_with(
            "anything", "obj.csv", artifact.path
        )

    Path(artifact.path).parent.mkdir(parents=True, exist_ok=True)
    Path(artifact.path).touch()
    assert Path(artifact.path).exists()
