import os
from dotenv import load_dotenv
from utils import KFPClientManager

load_dotenv(dotenv_path=".env")

def get_or_upload_pipeline(kfp_client, pipeline_yaml, pipeline_name, pipeline_version_name):
    pipeline_id = None
    version_id = None

    # Find pipeline_id by display_name
    pipelines_resp = kfp_client.list_pipelines(page_size=1000)
    pipelines = pipelines_resp.pipelines 
    for pipeline in pipelines:
        if getattr(pipeline, "display_name", None) == pipeline_name:
            pipeline_id = getattr(pipeline, "pipeline_id", None)
            break

    if pipeline_id:
        print(f"✅ Found existing pipeline: {pipeline_name} (id={pipeline_id})")
        # Find pipeline version by display_name
        versions_resp = kfp_client.list_pipeline_versions(pipeline_id=pipeline_id, page_size=100)
        versions = versions_resp.pipeline_versions 
        for version in versions:
            if getattr(version, "display_name", None) == pipeline_version_name:
                version_id = getattr(version, "pipeline_version_id", None)
                print(f"✅ Found existing version: {pipeline_version_name} (id={version_id})")
                break
        if not version_id:
            pv = kfp_client.upload_pipeline_version(
                pipeline_package_path=pipeline_yaml,
                pipeline_version_name=pipeline_version_name,
                pipeline_id=pipeline_id,
            )
            version_id = getattr(pv, "pipeline_version_id", None)
            print(f"⬆️  Uploaded new pipeline version: {pipeline_version_name} (id={version_id})")
    else:
        # Upload pipeline 
        pipeline = kfp_client.upload_pipeline(
            pipeline_package_path=pipeline_yaml,
            pipeline_name=pipeline_name,
            namespace="kubeflow-user-example-com" # if you define this arg none this will be a shared pipeline for all users ns
        )
        pipeline_id = getattr(pipeline, "pipeline_id", None)
        print(f"⬆️  Uploaded pipeline: {pipeline_name} (id={pipeline_id})")
        pipeline_version = kfp_client.upload_pipeline_version(
            pipeline_package_path=pipeline_yaml,
            pipeline_version_name=pipeline_version_name,
            pipeline_id=pipeline_id,
        )
        version_id = getattr(pipeline_version, "pipeline_version_id", None)
        print(f"⬆️  Uploaded pipeline version: {pipeline_version_name} (id={version_id})")

    return pipeline_id, version_id

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

    # 3️⃣ Define pipeline arguments
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
        "model_name": "xgb",
        "suffix": "underwriting",
        "experiment_name": "kfp",
    }

    pipeline_yaml = "pipeline.yaml"
    pipeline_name = "test pipeline"
    pipeline_version_name = "v1"

    # 4️⃣ Upload pipeline/version nếu cần và lấy id
    pipeline_id, version_id = get_or_upload_pipeline(
        kfp_client, pipeline_yaml, pipeline_name, pipeline_version_name
    )

    # 5️⃣ Get or create experiment
    namespace = os.getenv("KFP_NAMESPACE", "kubeflow-user-example-com")
    experiment = kfp_client.create_experiment(name="kfp", namespace=namespace)
    experiment_id = getattr(experiment, "experiment_id", None) or getattr(experiment, "id", None)

    # 6️⃣ Submit pipeline run
    run = kfp_client.run_pipeline(
        experiment_id=experiment_id,
        job_name="test job",
        pipeline_id=pipeline_id,
        version_id=version_id,
        params=pipeline_args,
    )
    print("🚀 Pipeline run submitted:", run)
