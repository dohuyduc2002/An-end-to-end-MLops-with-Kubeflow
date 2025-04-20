import kfp
import kfp.compiler as compiler
from utils import KFPClientManager
from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv(dotenv_path=".env")


if __name__ == "__main__":
    # 1️⃣ Compile the pipeline

    # 2️⃣ Setup KFP client
    kfp_client_manager = KFPClientManager(
        api_url=os.getenv("KFP_API_URL"),
        skip_tls_verify=os.getenv("KFP_SKIP_TLS_VERIFY", "False").lower() == "true",
        dex_username=os.getenv("KFP_DEX_USERNAME"),
        dex_password=os.getenv("KFP_DEX_PASSWORD"),
        dex_auth_type=os.getenv("KFP_DEX_AUTH_TYPE"),
    )
    kfp_client = kfp_client_manager.create_kfp_client()
    print("✅ Authenticated KFP client created successfully.")

    # 3️⃣ Define pipeline input parameters
    pipeline_arguments = {
        "minio_endpoint":    os.getenv("MINIO_ENDPOINT"),
        "minio_access_key":  os.getenv("MINIO_ACCESS_KEY"),
        "minio_secret_key":  os.getenv("MINIO_SECRET_KEY"),
        "bucket_name":       os.getenv("MINIO_BUCKET_NAME"),

        "train_object_name": "data/data/application_train.csv",
        "test_object_name":  "data/data/application_test.csv",

        "dest_train_object": "data/data/train/preprocessed_train.csv",
        "dest_test_object":  "data/data/test/preprocessed_test.csv",

        "n_features_to_select": "auto",
        "model_name":      "xgb",
        "version":         "v1",
        "experiment_name": "UnderwritingPipeline",
    }

    # 4️⃣ Submit the pipeline run
    run = kfp_client.create_run_from_pipeline_package(
        pipeline_file="pipeline.yaml",
        arguments=pipeline_arguments,
        run_name="Full Underwriting Pipeline Run",
        namespace="kubeflow-user-example-com",
    )
    print("🚀 Pipeline run submitted successfully:", run)
