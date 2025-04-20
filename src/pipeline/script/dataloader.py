from kfp import dsl
from kfp.dsl import Output, Dataset

@dsl.component(base_image="microwave1005/scipy-img:latest")
def minio_data_loader_component(
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    bucket_name: str,
    object_name: str,
    output: Output[Dataset]
):
    """
    Downloads a file from MinIO and saves it to the output artifact.
    
    Parameters:
      - minio_endpoint: The endpoint of the MinIO server.
      - minio_access_key: The MinIO access key.
      - minio_secret_key: The MinIO secret key.
      - bucket_name: Name of the bucket containing the file.
      - object_name: Name of the file to download.
      - output: An output artifact where the downloaded file will be stored.
    """
    import os
    from minio import Minio

    client = Minio(
        minio_endpoint,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        secure=False  # Change to True if using HTTPS.
    )
    os.makedirs(os.path.dirname(output.path), exist_ok=True)
    client.fget_object(bucket_name, object_name, output.path)
    print(f"Downloaded {object_name} from bucket {bucket_name} to {output.path}")

if __name__ == "__main__":
    import kfp.compiler as compiler
    # Compile the component into YAML and save it to the yaml_components directory.
    compiler.Compiler().compile(minio_data_loader_component, "../yaml_components/dataloader.yaml")
