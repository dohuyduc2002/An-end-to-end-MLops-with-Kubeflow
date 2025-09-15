from kfp import dsl
from kfp.components import load_component_from_file
from pathlib import Path

COMP_DIR = (Path(__file__).parent / "scripts" / "component_metadata").resolve()

spark_sa_op = load_component_from_file(COMP_DIR / "apply_rbac_spark.yaml")
silver_layer_op = load_component_from_file(COMP_DIR / "silver_layer_etl.yaml")
gold_layer_op = load_component_from_file(COMP_DIR / "gold_layer_etl.yaml")
notify_slack_op = load_component_from_file(COMP_DIR / "slack_notification.yaml")


@dsl.pipeline(
    name="ETL_Pipeline",
    description="ETL Pipeline with spark"
)
def etl_pipeline(
    namespace: str,
    application_silver_manifest: str,
    merged_silver_manifest: str,
    data_mart_gold_manifest: str,
    spark_sa_manifest: str,
    slack_channel: str,
    slack_bot_token: str,
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    minio_bucket: str,
):
    spark_sa = (
        spark_sa_op(
            namespace=namespace,
            minio_endpoint=minio_endpoint,
            minio_access_key=minio_access_key,
            minio_secret_key=minio_secret_key,
            minio_bucket=minio_bucket,
            yaml_object=spark_sa_manifest,
        )
        .set_display_name("Setup Spark RBAC")
        .set_caching_options(enable_caching=True)
    )
    
    application_silver_etl = (
        silver_layer_op(
        namespace = namespace,
        minio_endpoint = minio_endpoint,
        minio_access_key = minio_access_key,
        minio_secret_key = minio_secret_key,
        minio_bucket = minio_bucket,
        yaml_object = application_silver_manifest
    )
        .set_display_name("silver: application")
        .set_caching_options(enable_caching=False)
    ).after(spark_sa)

    merged_silver_etl = (
        silver_layer_op(
        namespace = namespace,
        minio_endpoint = minio_endpoint,
        minio_access_key = minio_access_key,
        minio_secret_key = minio_secret_key,
        minio_bucket = minio_bucket,
        yaml_object = merged_silver_manifest
    )
        .set_display_name("silver: merged")
        .set_caching_options(enable_caching=False)
    ).after(spark_sa)
    
    data_mart_gold_etl = (
        gold_layer_op(
            namespace = namespace,
            minio_endpoint = minio_endpoint,
            minio_access_key = minio_access_key,
            minio_secret_key = minio_secret_key,
            minio_bucket = minio_bucket,
            yaml_object = data_mart_gold_manifest,
            application_silver_status = application_silver_etl.outputs["final_status"],
            merged_silver_status = merged_silver_etl.outputs["final_status"]
        )
        .set_display_name("gold: data_mart")
        .set_caching_options(enable_caching=False)
        .after(application_silver_etl, merged_silver_etl)
    )

    
    notify_slack_op(
        slack_channel=slack_channel,
        message="ETL Pipeline completed successfully",
        slack_bot_token=slack_bot_token
    ).after(data_mart_gold_etl)
