from pyspark.sql import SparkSession
from pyspark.sql import functions as F

import sys
from pathlib import Path

from silver_etl_schema import (
    silver_fact_bureau_balance_schema,
    silver_dim_bureau_schema,
)
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    MINIO_ENDPOINT,
    MINIO_ACCESS_KEY,
    MINIO_SECRET_KEY,
    SILVER_PATH_MERGED,
    BRONZE_PATH_MERGED,
    write_with_uc_sql,
    add_scd_type2_cols
)




def main():
    spark = (
        SparkSession.builder
        .config("spark.hadoop.fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem")
        .config("spark.hadoop.fs.AbstractFileSystem.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS")
        .config("spark.hadoop.google.cloud.auth.service.account.enable", "true")
        .config("spark.hadoop.google.cloud.auth.service.account.json.keyfile", "/var/secrets/gcp/gcp-key.json")
        .config("spark.hadoop.fs.s3a.endpoint", f"http://{MINIO_ENDPOINT}")
        .config("spark.hadoop.fs.s3a.access.key", MINIO_ACCESS_KEY)
        .config("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET_KEY)
        .appName("Gold merge batching")
        .getOrCreate()
    )

    bronze_bureau_balance_table = spark.read.format("delta").load(BRONZE_PATH_MERGED)
    silver_bureau_balance_table = bronze_bureau_balance_table.select(
        "sk_id_bureau",
        "sk_id_curr",
        "months_balance",
        "status",
        "updated",
        "event_ts")

    bureau_table = (
        spark.read.format("csv")
        .option("header", "true")
        .load("s3a://sample-data/bureau.csv")
    )

    for col in bureau_table.columns:
        bureau_table = bureau_table.withColumnRenamed(col, col.lower())
    
    silver_bureau_table_scd_type2 = add_scd_type2_cols(bureau_table, ts_col=None)
    silver_bureau_table_scd2 = silver_bureau_table_scd_type2.withColumn("event_ts", F.current_timestamp())
        
    spark.sql("""
    CREATE SCHEMA IF NOT EXISTS silver
    LOCATION 'gs://unity-catalog-dhduc/silver'
    """)

    # Ghi dim_bureau
    write_with_uc_sql(
        spark,
        silver_bureau_table_scd2,
        layer="silver",
        table_name="dim_bureau",
        schema=silver_dim_bureau_schema,      # vẫn truyền để giữ interface, nhưng thực tế df quyết định
        base_path=SILVER_PATH_MERGED,
        mode="append"
    )

    # Ghi fact_bureau_balance
    write_with_uc_sql(
        spark,
        silver_bureau_balance_table,
        layer="silver",
        table_name="fact_bureau_balance",
        schema=silver_fact_bureau_balance_schema,
        base_path=SILVER_PATH_MERGED,
        mode="append"
    )

    
if __name__ == "__main__":
    main()