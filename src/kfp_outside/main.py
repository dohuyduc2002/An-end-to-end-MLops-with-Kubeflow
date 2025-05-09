# main.py
import os
import kfp
from dotenv import load_dotenv
from utils import KFPClientManager

# Load .env only to populate MinIO creds & bucket
load_dotenv(dotenv_path=".env")

if __name__ == "__main__":
    # 1️⃣ Create authenticated KFP client
    client_mgr = KFPClientManager(
        api_url=os.getenv("KFP_API_URL"),
        dex_username=os.getenv("KFP_DEX_USERNAME"),
        dex_password=os.getenv("KFP_DEX_PASSWORD"),
        dex_auth_type=os.getenv("KFP_DEX_AUTH_TYPE", "local"),
        skip_tls_verify=os.getenv("KFP_SKIP_TLS_VERIFY", "False").lower() == "true",
    )
    kfp_client = client_mgr.create_kfp_client()
    print("✅ Authenticated KFP client created.")

    # 2️⃣ Read MinIO settings from env
    minio_endpoint   = os.environ["MINIO_ENDPOINT"]
    minio_access_key = os.environ["MINIO_ACCESS_KEY"]
    minio_secret_key = os.environ["MINIO_SECRET_KEY"]
    bucket_name      = os.environ["MINIO_BUCKET_NAME"]

    # 3️⃣ Define the rest of pipeline parameters inline
    pipeline_args = {
        "minio_endpoint":       minio_endpoint,
        "minio_access_key":     minio_access_key,
        "minio_secret_key":     minio_secret_key,
        "bucket_name":          bucket_name,
        "raw_train_object":     "data/application_train.csv",
        "raw_test_object":      "data/application_test.csv",
        "dest_train_object":    "data/train/preprocessed_train.csv",
        "dest_test_object":     "data/test/preprocessed_test.csv",
        "n_features_to_select": "auto",
        "data_version":         "v1",
        "model_name":           "xgb",
        "version":              "v1_xgb",
        "experiment_name":      "Underwriting-model",
    }

    # 4️⃣ Submit the pipeline run using existing pipeline.yaml
    run = kfp_client.create_run_from_pipeline_package(
        pipeline_file="pipeline.yaml",
        arguments=pipeline_args,
        run_name="Underwriting Full Run",
        namespace=os.getenv("KFP_NAMESPACE", "kubeflow-user-example-com"),
    )
    print("🚀 Pipeline run submitted:", run)
