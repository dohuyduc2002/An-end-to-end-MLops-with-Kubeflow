# pipeline.py
from kfp import dsl
from kfp.components import load_component_from_file

dataloader_op = load_component_from_file("components/dataloader.yaml")
preprocess_op = load_component_from_file("components/preprocess.yaml")
modeling_op   = load_component_from_file("components/model.yaml")

@dsl.pipeline(
    name="UnderwritingWorkflow",
    description="Download raw → preprocess → download processed → train & register",
)
def underwriting_pipeline(
    minio_endpoint:       str,
    minio_access_key:     str,
    minio_secret_key:     str,
    bucket_name:          str,
    raw_train_object:     str,
    raw_test_object:      str,
    dest_train_object:    str = "processed/train.csv",
    dest_test_object:     str = "processed/test.csv",
    n_features_to_select: str = "auto",
    data_version:         str = "v1",
    model_name:           str = "xgb",
    version:              str = "v1",
    experiment_name:      str = "UnderwritingPipeline",
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
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        bucket_name=bucket_name,
        dest_train_object=dest_train_object,
        dest_test_object=dest_test_object,
        n_features_to_select=n_features_to_select,
        data_version=data_version,
    ).after(raw_te)

    # 4️⃣ Download processed train
    proc_tr = dataloader_op(
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        bucket_name=bucket_name,
        object_name=prep.outputs["train_key"],
    ).after(prep)

    # 5️⃣ Download processed test
    proc_te = dataloader_op(
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        bucket_name=bucket_name,
        object_name=prep.outputs["test_key"],
    ).after(prep)

    # 6️⃣ Modeling
    modeling_op(
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        train_csv=proc_tr.outputs["output"],
        test_csv= proc_te.outputs["output"],
        model_name=model_name,
        version=version,
        experiment_name=experiment_name,
    ).after(proc_te)


if __name__ == "__main__":
    import kfp.compiler as compiler
    compiler.Compiler().compile(
        underwriting_pipeline,
        "pipeline.yaml",
    )
