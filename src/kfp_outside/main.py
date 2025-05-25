# main.py
import os
from dotenv import load_dotenv
from utils import KFPClientManager

load_dotenv(dotenv_path=".env")

if __name__ == "__main__":
    # 1️⃣ Create authenticated KFP client
    client_auth_manager = KFPClientManager(
        api_url=os.getenv("KFP_API_URL"),
        dex_username=os.getenv("KFP_DEX_USERNAME"),
        dex_password=os.getenv("KFP_DEX_PASSWORD"),
        dex_auth_type=os.getenv("KFP_DEX_AUTH_TYPE", "local"),
        skip_tls_verify=os.getenv("KFP_SKIP_TLS_VERIFY", "False").lower() == "true",
    )
    kfp_client = client_auth_manager.create_kfp_client()
    print("✅ Authenticated KFP client created.")

    # 2️⃣ Read MinIO settings from env
    minio_endpoint   = os.environ["MINIO_ENDPOINT"]
    minio_access_key = os.environ["MINIO_ACCESS_KEY"]
    minio_secret_key = os.environ["MINIO_SECRET_KEY"]
    bucket_name      = os.environ["MINIO_BUCKET_NAME"]
    mlflow_endpoint  = os.environ["MLFLOW_ENDPOINT"]

    # 3️⃣ Define the rest of pipeline parameters inline
    pipeline_args = {
        "minio_endpoint": minio_endpoint,
        "minio_access_key": minio_access_key,
        "minio_secret_key": minio_secret_key,
        "bucket_name": bucket_name,
        "mlflow_endpoint": mlflow_endpoint,
        "raw_train_object": "data/application_train.csv",
        "raw_test_object": "data/application_test.csv",
        "dest_train_object": "preprocessed_train.csv",
        "dest_test_object": "preprocessed_test.csv",
        "parent_run_name": "xgb_experiment_optuna_search",
        "n_features_to_select": "auto",
        "data_version": "v1",
        "model_name": "xgb",  # xgb or lgbm only
        "suffix": "underwriting",
        "experiment_name": "kfp",
    }

    # 4️⃣ Submit the pipeline run using existing pipeline.yaml
    run = kfp_client.create_run_from_pipeline_package(
        pipeline_file="pipeline.yaml",
        arguments=pipeline_args,
        run_name="Underwriting Full Run",
        namespace=os.getenv("KFP_NAMESPACE", "kubeflow-user-example-com"),
    )
    print("🚀 Pipeline run submitted:", run)
