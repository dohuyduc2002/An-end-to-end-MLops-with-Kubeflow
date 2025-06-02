import os
from dotenv import load_dotenv
from utils import KFPClientManager, get_or_upload_pipeline

load_dotenv(dotenv_path=".env")



if __name__ == "__main__":
    client_auth_manager = KFPClientManager(
        api_url=os.getenv("KFP_API_URL"),
        dex_username=os.getenv("KFP_DEX_USERNAME"),
        dex_password=os.getenv("KFP_DEX_PASSWORD"),
        dex_auth_type="local",
        skip_tls_verify=True,
    )
    kfp_client = client_auth_manager.create_kfp_client()
    print("✅ Authenticated KFP client created.")

    # Read MinIO settings from env
    minio_endpoint = os.environ["MINIO_ENDPOINT"]
    minio_access_key = os.environ["MINIO_ACCESS_KEY"]
    minio_secret_key = os.environ["MINIO_SECRET_KEY"]
    bucket_name = os.environ["MINIO_BUCKET_NAME"]
    mlflow_endpoint = os.environ["MLFLOW_ENDPOINT"]

    # Define pipeline arguments
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
        "parent_run_name": "xgb_optuna_search",
        "n_features_to_select": "auto",
        "data_version": "v1",
        "model_name": "xgb", #xgb or lgbm
        "suffix": "underwriting",
        "experiment_name": "Kubeflow Pipeline outside",
    }

    pipeline_yaml = "pipeline.yaml"
    pipeline_name = "kfp-outside-pipeline"  # due to my code, the 1st version will be uploaded with this name and version_name
    version_name = "v1"  # this version will be a reference for recurring runs in cicd

    # Upload pipeline/version and get IDs
    pipeline_id, version_id, version_name = get_or_upload_pipeline(
        kfp_client, pipeline_yaml, pipeline_name, version_name
    )

    namespace = os.getenv("KFP_NAMESPACE")
    experiment = kfp_client.create_experiment(name="kfp_outside_cluster", namespace=namespace)
    experiment_id = getattr(experiment, "experiment_id")

    run = kfp_client.run_pipeline(
        experiment_id=experiment_id,
        job_name="Underwriting Model Job Run",
        pipeline_id=pipeline_id,
        version_id=version_id,
        params=pipeline_args,
    )
    print("🚀 Pipeline run submitted:", run)
