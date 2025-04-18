import kfp
import kfp.compiler as compiler
from utils import KFPClientManager

if __name__ == "__main__":
    # 1️⃣ Compile the pipeline

    # 2️⃣ Setup KFP client
    kfp_client_manager = KFPClientManager(
        api_url="http://localhost:8080/pipeline",
        skip_tls_verify=True,
        dex_username="user@example.com",
        dex_password="12341234",
        dex_auth_type="local",
    )
    kfp_client = kfp_client_manager.create_kfp_client()
    print("✅ Authenticated KFP client created successfully.")

    # 3️⃣ Define pipeline input parameters
    pipeline_arguments = {
        "minio_endpoint":    "minio-service.kubeflow.svc.cluster.local:9000",
        "minio_access_key":  "minio",
        "minio_secret_key":  "minio123",
        "bucket_name":       "sample-data",

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
