from kfp import dsl
from kfp.dsl import Output, Dataset
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import SCIPY_IMAGE

@dsl.component(base_image=SCIPY_IMAGE
            )
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

