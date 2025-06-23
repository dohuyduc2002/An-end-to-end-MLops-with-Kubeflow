from kfp import dsl
from kfp.dsl import Output, Dataset
from pathlib import Path
from component_utils import BASE_IMAGE, TARGET_IMAGE


@dsl.component(base_image=BASE_IMAGE, target_image=TARGET_IMAGE)
def dataloader(
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    bucket_name: str,
    object_name: str,
    output: Output[Dataset],   
):
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

