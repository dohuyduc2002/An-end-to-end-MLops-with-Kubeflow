import os
from pathlib import Path
from dotenv import load_dotenv
from minio import Minio


def test_dataloader_like_real(tmp_path):
    # ✅ Load .env file from the project root directory
    root_dir = Path(__file__).resolve().parents[1]  
    dotenv_path = root_dir / ".env"
    load_dotenv(dotenv_path)

    # Read environment variables
    minio_endpoint = "localhost:9000"  # Use port-forwarded endpoint for local testing
    minio_access_key = os.getenv("MINIO_ACCESS_KEY")
    minio_secret_key = os.getenv("MINIO_SECRET_KEY")
    bucket_name = "sample-data"
    object_name = "data/data/application_train.csv"

    # Simulate the output.path of a KFP Dataset artifact
    output_path = tmp_path / "artifact" / "application_train.csv"
    os.makedirs(output_path.parent, exist_ok=True)

    # Download file from MinIO
    client = Minio(
        minio_endpoint,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        secure=False,
    )
    client.fget_object(bucket_name, object_name, str(output_path))

    # Validate that the file was downloaded successfully
    assert output_path.exists(), "❌ File not downloaded"
    assert output_path.stat().st_size > 0, "❌ Downloaded file is empty"
