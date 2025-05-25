# pipeline.py
from kfp import dsl
from kfp.components import load_component_from_file
from pathlib import Path

COMP_DIR = Path(__file__).with_suffix("").parent / "components"

dataloader_op = load_component_from_file(COMP_DIR / "dataloader.yaml")
preprocess_op = load_component_from_file(COMP_DIR / "preprocess.yaml")
modeling_op   = load_component_from_file(COMP_DIR / "model.yaml")


@dsl.pipeline(
    name="UnderwritingWorkflow",
    description="Download raw → preprocess → download processed → train & register",
)
def underwriting_pipeline(
    minio_endpoint: str,
    mlflow_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    bucket_name: str,
    raw_train_object: str,
    raw_test_object: str,
    parent_run_name: str,
    dest_train_object: str,
    dest_test_object: str,
    n_features_to_select: str,
    data_version: str,
    model_name: str,
    suffix: str,
    experiment_name: str,
):
    # 1️⃣ Download raw train
    raw_tr = dataloader_op(
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        bucket_name=bucket_name,
        object_name=raw_train_object,
    )

    # 2️⃣ Download raw test
    raw_te = dataloader_op(
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        bucket_name=bucket_name,
        object_name=raw_test_object,
    )

    # 3️⃣ Preprocess
    prep = preprocess_op(
        train_csv=raw_tr.outputs["output"],
        test_csv= raw_te.outputs["output"],
        mlflow_endpoint=mlflow_endpoint,
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        experiment_name=experiment_name,
        parent_run_name=parent_run_name,
        dest_train_object=dest_train_object,
        dest_test_object=dest_test_object,
        n_features_to_select=n_features_to_select,
        data_version=data_version,
    ).after(raw_te)

    # 6️⃣ Modeling
    modeling_op(
        mlflow_endpoint=mlflow_endpoint,
        train_csv=prep.outputs["output_train_csv"],
        test_csv=prep.outputs["output_test_csv"],
        mlflow_run_id=prep.outputs["mlflow_run_id"],
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        model_name=model_name,
        suffix=suffix,
        experiment_name=experiment_name,
    ).after(prep)


if __name__ == "__main__":
    import kfp.compiler as compiler
    compiler.Compiler().compile(
        underwriting_pipeline,
        "pipeline.yaml",
    )
