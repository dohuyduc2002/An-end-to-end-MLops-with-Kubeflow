from minio import Minio
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables
BASE_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# MinIO client setup
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

# List of MinIO paths (prefixes) to download completely
prefixes = [
    {"bucket": "mlflow", "prefix": "1/b48e75a372e4462bbc627ebed1a75841/artifacts/model/"},
    {"bucket": "mlpipeline", "prefix": "v2/artifacts/underwritingworkflow/2770a781-fdbb-48a1-af71-fb112604579d/dataloader-3/1e25a48d-adee-4e52-b92b-7a339b255e3a/"},
    {"bucket": "mlpipeline", "prefix": "v2/artifacts/underwritingworkflow/2770a781-fdbb-48a1-af71-fb112604579d/dataloader-4/757989d7-a7fd-4f3d-a250-83b4fda91ebd/"},
    {"bucket": "mlpipeline", "prefix": "v2/artifacts/underwritingworkflow/2770a781-fdbb-48a1-af71-fb112604579d/modeling/3a4a6346-3abc-474c-95a0-4a7c96a4f9c1/"},
    {"bucket": "mlpipeline", "prefix": "v2/artifacts/underwritingworkflow/2770a781-fdbb-48a1-af71-fb112604579d/preprocess/bdce64f1-17f8-4ace-bc4e-8b298a3b8c87/"}
]

# Local base directory
local_root = Path("joblib")
local_root.mkdir(exist_ok=True)

# Function to download all objects under a prefix
def download_prefix(bucket: str, prefix: str):
    print(f"Listing objects in bucket '{bucket}' with prefix '{prefix}'...")
    for obj in client.list_objects(bucket, prefix=prefix, recursive=True):
        try:
            relative_path = Path(obj.object_name).relative_to(prefix)  # relative to the prefix
        except ValueError:
            print(f"Skipping object '{obj.object_name}' as it does not start with the prefix '{prefix}'")
            continue
        local_path = local_root / bucket / prefix / relative_path  # recreate full path

        # Ensure parent directories exist
        local_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {obj.object_name} to {local_path}...")
        client.fget_object(bucket, obj.object_name, str(local_path))
        print(f"Saved {local_path}")

# Start downloading
for item in prefixes:
    download_prefix(item["bucket"], item["prefix"])
