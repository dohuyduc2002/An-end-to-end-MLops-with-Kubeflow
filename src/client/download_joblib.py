from minio import Minio
from pathlib import Path
from dotenv import load_dotenv
import os

BASE_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

minio_endpoint = 'localhost:9000'
access_key = os.getenv("MINIO_ACCESS_KEY")
secret_key = os.getenv("MINIO_SECRET_KEY")
secure = False

client = Minio(
    minio_endpoint,
    access_key=access_key,
    secret_key=secret_key,
    secure=secure
)

files = [
    {
        "bucket": "mlpipeline",
        "object_name": "v2/artifacts/underwritingworkflow/c33b21bd-b408-417d-b7b1-71361b23a889/preprocess/d6f6a45e-6a94-47f0-8598-cc83a2bbdf7a/transformer_joblib",
        "filename": "transformer.joblib"
    },
    {
        "bucket": "mlpipeline",
        "object_name": "v2/artifacts/underwritingworkflow/c33b21bd-b408-417d-b7b1-71361b23a889/modeling/32fda081-3ee1-43a0-a083-5735c7366978/model_joblib",
        "filename": "model.joblib"
    }
]

Path("joblib").mkdir(exist_ok=True)
for f in files:
    local_path = f"joblib/{f['filename']}"
    print(f"Downloading {f['filename']}...")
    client.fget_object(f["bucket"], f["object_name"], local_path)
    print(f"Saved to {local_path}")


