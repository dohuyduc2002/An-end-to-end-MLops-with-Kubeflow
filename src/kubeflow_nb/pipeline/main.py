import os
from dotenv import load_dotenv
from utils import upload_pipeline
from kfp import Client

load_dotenv()

if __name__ == "__main__":

    # Read MinIO settings from env
    minio_endpoint = os.environ["MINIO_ENDPOINT"]
    minio_access_key = os.environ["MINIO_ACCESS_KEY"]
    minio_secret_key = os.environ["MINIO_SECRET_KEY"]
    bucket_name = os.environ["MINIO_BUCKET_NAME"]
    mlflow_endpoint = os.environ["MLFLOW_ENDPOINT"]
    namespace = os.getenv("KFP_NAMESPACE")

    kfp_client = Client()  
    print("✅ Authenticated KFP client created.")

    # Define pipeline arguments
    pipeline_args = {
        "minio_endpoint": minio_endpoint,
        "minio_access_key": minio_access_key,
        "minio_secret_key": minio_secret_key,
        "bucket_name": bucket_name,
        "mlflow_endpoint": mlflow_endpoint,
        "raw_train_object": "data/application_train.csv",
        "raw_test_object": "data/application_test.csv",
        "parent_run_name": "xgb_optuna_search",
        "n_features_to_select": "auto",
        "model_name": "xgb", #xgb or lgbm
        "suffix": "underwrite",
        "experiment_name": "Underwriting_kfp",
    }

    pipeline_yaml = "pipeline.yaml"
    pipeline_name = "xgb_model"  # due to my code, the 1st version will be uploaded with this name and version_name
    version_name = "v1"  # this version will be a reference for recurring runs in cicd

    # Upload pipeline/version and get IDs
    pipeline_id, version_id, version_name = upload_pipeline(
        kfp_client, pipeline_yaml, pipeline_name, version_name
    )

    experiment = kfp_client.create_experiment(name="underwrite_experiment", namespace=namespace)
    experiment_id = getattr(experiment, "experiment_id")

    run = kfp_client.run_pipeline(
        experiment_id=experiment_id,
        job_name="Underwriting Model Job Run",
        pipeline_id=pipeline_id,
        version_id=version_id,
        params=pipeline_args,
    )
    print("🚀 Pipeline run submitted:", run)
