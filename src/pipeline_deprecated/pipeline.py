from kfp import dsl
from kfp.components import load_component_from_file

preprocess_op = load_component_from_file("yaml_components/preprocess_and_push.yaml")
train_op      = load_component_from_file("yaml_components/trainer.yaml")

@dsl.pipeline(name="Preprocess + Train", description="Pipeline with MinIO-backed CSV exchange.")
def pipeline(
    minio_endpoint:    str,
    minio_access_key:  str,
    minio_secret_key:  str,
    bucket_name:       str,
    train_object_name: str,
    test_object_name:  str,
    dest_train_object: str = "data/data/train/preprocessed_train.csv",
    dest_test_object:  str = "data/data/test/preprocessed_test.csv",
    n_features_to_select: str = "auto",
    model_name:        str = "xgb",
    version:           str = "v1",
    experiment_name:   str = "UnderwritingPipeline",
):
    preprocess_task = preprocess_op(
        minio_endpoint       = minio_endpoint,
        minio_access_key     = minio_access_key,
        minio_secret_key     = minio_secret_key,
        bucket_name          = bucket_name,
        train_object_name    = train_object_name,
        test_object_name     = test_object_name,
        dest_train_object    = dest_train_object,
        dest_test_object     = dest_test_object,
        n_features_to_select = n_features_to_select,
        data_version         = version,
    )

    train_op(
        minio_endpoint      = minio_endpoint,
        minio_access_key    = minio_access_key,
        minio_secret_key    = minio_secret_key,
        bucket_name         = bucket_name,
        processed_train_key = preprocess_task.outputs["train_key"],
        processed_test_key  = preprocess_task.outputs["test_key"],
        model_name          = model_name,
        version             = version,
        experiment_name     = experiment_name,
    ).after(preprocess_task) \
    # .set_env_variable("AWS_ACCESS_KEY_ID", minio_access_key) \
    # .set_env_variable("AWS_SECRET_ACCESS_KEY", minio_secret_key)
 


# ⬇️ Add this to allow standalone compilation
if __name__ == "__main__":
    import kfp.compiler as compiler
    compiler.Compiler().compile(pipeline, "pipeline.yaml")
    
