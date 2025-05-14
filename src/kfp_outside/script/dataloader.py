# scripts/dataloader.py
from kfp import dsl
from kfp.dsl import Output, Dataset

@dsl.component(base_image="microwave1005/scipy-img:latest",
                packages_to_install=[
                "protobuf==4.25.5",
                "kfp==2.12.1",
                "fastapi==0.104.1",
                "uvicorn[standard]==0.24.0",
                "loguru==0.7.2",
                "joblib==1.3.2",
                "pandas==2.1.3",
                "pytest==7.4.3",
                "numpy==1.24.4",
                "mlflow==2.8.1",
                "matplotlib==3.8.1",
                "pydantic==1.10.8",
                "ortools==9.7.2996",
                "requests==2.31.0",
                "boto3",
                "shap",
                "optuna",
                "optbinning",
                "urllib3",
                "minio",
                "lightgbm",
                "python-dotenv"
            ])
def dataloader(
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    bucket_name: str,
    object_name: str,
    output: Output[Dataset],   
):
    """
    Download a single object from MinIO into a KFP Dataset artifact.
    """
    from minio import Minio
    import os

    os.makedirs(os.path.dirname(output.path), exist_ok=True)
    client = Minio(
        minio_endpoint,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        secure=False,
    )
    client.fget_object(bucket_name, object_name, output.path)
    print(f"Downloaded {object_name} to {output.path}")

if __name__ == "__main__":
    from pathlib import Path
    import kfp.compiler as compiler

    # Define paths using pathlib
    current_dir = Path(__file__).parent
    components_dir = current_dir.parent / "components"
    components_dir.mkdir(parents=True, exist_ok=True)

    # Compile and write the YAML to the components directory
    compiler.Compiler().compile(
        dataloader,
        str(components_dir / "dataloader.yaml"),
    )

