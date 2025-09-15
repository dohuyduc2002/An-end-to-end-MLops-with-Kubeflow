from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Artifact
from component_utils import BASE_IMAGE, TARGET_IMAGE

@dsl.component(base_image=BASE_IMAGE, target_image=TARGET_IMAGE)
def silver_layer_etl(
    submitted_spec: Output[Artifact],
    final_status: Output[Dataset],
    namespace: str,
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    minio_bucket: str,
    yaml_object: str,
):
    import yaml
    from kubernetes import client, config
    from minio import Minio
    import io
    
    
    config.load_incluster_config()
    k8s_api = client.CustomObjectsApi()
    
    minio_client = Minio(
        minio_endpoint,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        secure=False 
    )

    # Download YAML file from MinIO into memory
    response = minio_client.get_object(minio_bucket, yaml_object)
    yaml_bytes = response.read()
    
    # read yaml
    spark_app = yaml.safe_load(io.BytesIO(yaml_bytes))
    
    with open(submitted_spec.path, "w") as f:
        yaml.safe_dump(spark_app, f, sort_keys=False)
        
    name = spark_app["metadata"]["name"]
        
    k8s_api.create_namespaced_custom_object(
        group="sparkoperator.k8s.io",
        version="v1beta2",
        namespace=namespace,
        plural="sparkapplications",
        body=spark_app,
    )
    
    last_status = {}
    state = None
    
    while True:
        app = k8s_api.get_namespaced_custom_object(
            group="sparkoperator.k8s.io",
            version="v1beta2",
            namespace=namespace,
            plural="sparkapplications",
            name=name,
        )
        status = app.get("status", {})
        last_status = status
        state = status.get("applicationState", {}).get("state")
        print(f"[INFO] {name} state: {state}")
        
        if state in ("COMPLETED", "FAILED"):
            break

    # Write the final status to the output
    with open(final_status.path, "w") as f:
        yaml.safe_dump(last_status, f, sort_keys=False)
        
    if state != "COMPLETED":
        raise RuntimeError(f"SparkApplication ended with state={state}")