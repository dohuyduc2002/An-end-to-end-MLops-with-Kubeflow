import os
from pyspark.sql import functions as F

def write_with_delta_sql(spark, df, table_name, schema, base_path, table_ddl, mode="append"):
    table_full_name = f"{schema}.{table_name}"
    table_location = f"{base_path.rstrip('/')}/{table_name}"

    if mode == "overwrite":
        spark.sql(f"DROP TABLE IF EXISTS {table_full_name}")
        spark.sql(f"""
            CREATE TABLE {table_full_name} (
                {table_ddl.toDDL()}
            )
            USING delta
            LOCATION '{table_location}'
        """)
        df.write.format("delta").mode("overwrite").save(table_location)
    else:  
        spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {table_full_name} (
                {table_ddl.toDDL()}
            )
            USING delta
            LOCATION '{table_location}'
        """)
        df.write.format("delta").mode("append").save(table_location)


def add_scd_type2_cols(df, ts_col):
    ts = F.col(ts_col).cast("timestamp") if ts_col else F.current_timestamp()
    return (df
        .withColumn("is_current", F.lit(True))
        .withColumn("effective_date", ts)
        .withColumn("end_date", F.to_timestamp(F.lit("2025-12-31 23:59:59")))
    )
    
CLICKHOUSE_HOST = os.getenv("CLICKHOUSE_HOST")
CLICKHOUSE_USER = os.getenv("CLICKHOUSE_USER")
CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD")
CLICKHOUSE_DATABASE = os.getenv("CLICKHOUSE_DATABASE")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS")


SCHEMA_REGISTRY_CONFIG = {
    "url": os.getenv("SCHEMA_REGISTRY_URL")
}

APPLICATION_TOPIC = os.getenv("APPLICATION_TOPIC")
MERGED_TOPIC = os.getenv("MERGED_TOPIC")

BRONZE_PATH_APPLICATION = os.getenv("BRONZE_PATH_APPLICATION")
CHECKPOINT_PATH_APPLICATION = os.getenv("CHECKPOINT_PATH_APPLICATION")

BRONZE_PATH_MERGED = os.getenv("BRONZE_PATH_MERGED")
CHECKPOINT_PATH_MERGED = os.getenv("CHECKPOINT_PATH_MERGED")

SILVER_PATH_APPLICATION = os.getenv("SILVER_PATH_APPLICATION")
SILVER_PATH_MERGED = os.getenv("SILVER_PATH_MERGED")

DATA_MART_GOLD_PATH = os.getenv("DATA_MART_GOLD_PATH")


# BRONZE_PATH_APPLICATION = "gs://unity-catalog-dhduc/bronze/application"
# CHECKPOINT_PATH_APPLICATION = "gs://unity-catalog-dhduc/checkpoint/application"

# BRONZE_PATH_MERGED = "gs://unity-catalog-dhduc/bronze/merged"
# CHECKPOINT_PATH_MERGED = "gs://unity-catalog-dhduc/checkpoint/merged"

# SILVER_PATH_APPLICATION = "gs://unity-catalog-dhduc/silver"
# SILVER_PATH_MERGED = "gs://unity-catalog-dhduc/silver"

# DATA_MART_GOLD_PATH = "gs://unity-catalog-dhduc/data-mart"