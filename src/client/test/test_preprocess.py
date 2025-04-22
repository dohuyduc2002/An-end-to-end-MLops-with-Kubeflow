import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import joblib
from minio import Minio
from loguru import logger

# Add src/ to sys.path so that kfp.script can be imported
sys.path.append(str(Path(__file__).resolve().parents[2]))

def test_preprocess_real(tmp_path):
    # Load .env from kfp/
    project_root = Path(__file__).resolve().parents[2]
    dotenv_path = project_root / "kfp" / ".env"
    load_dotenv(dotenv_path)

    # MinIO config
    minio_endpoint = "localhost:9000"
    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")
    bucket = "sample-data"

    # Input object keys
    train_key = "data/data/application_train.csv"
    test_key = "data/data/application_test.csv"
    version = "v1"

    # MinIO client
    client = Minio(minio_endpoint, access_key=access_key, secret_key=secret_key, secure=False)

    # Download input CSVs to temp dir
    local_train = tmp_path / "train.csv"
    local_test = tmp_path / "test.csv"
    client.fget_object(bucket, train_key, str(local_train))
    client.fget_object(bucket, test_key, str(local_test))

    transformer_path = project_root / "client" / "joblib" / "transformer.joblib"
    transformer_path.parent.mkdir(parents=True, exist_ok=True)
    output_model = type("OutputMock", (), {"path": str(transformer_path)})()

    # Import and call real logic
    from kfp.script.preprocess import preprocess
    tr_key, te_key = preprocess.python_func(
        train_csv=local_train,
        test_csv=local_test,
        transformer_joblib=output_model,
        minio_endpoint=minio_endpoint,
        minio_access_key=access_key,
        minio_secret_key=secret_key,
        bucket_name=bucket,
        dest_train_object="data/data/train/preprocessed_train.csv",
        dest_test_object="data/data/test/preprocessed_test.csv",
        n_features_to_select="auto",
        data_version=version,
    )


    # === Assert outputs ===
    assert tr_key.endswith(f"_{version}.csv")
    assert te_key.endswith(f"_{version}.csv")
    assert client.stat_object(bucket, tr_key).size > 0
    assert client.stat_object(bucket, te_key).size > 0
    assert transformer_path.exists(), "❌ transformer.joblib not found"
    transformer_obj = joblib.load(transformer_path)
    assert "binning_process" in transformer_obj and "selector" in transformer_obj, "❌ Invalid transformer format"

    # === Clean up 
    client.remove_object(bucket, tr_key)
    client.remove_object(bucket, te_key)
    logger.info("Deleted test and train objects from MinIO after test")
    transformer_path.unlink(missing_ok=True)
    os.remove(transformer_path)
    logger.info("transformer.joblib deleted after test")
