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
    write_with_delta_sql,
    add_scd_type2_cols
)

def main():
    spark = (
        SparkSession.builder
        .appName("Silver bureau batching")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("hive.metastore.uris", "thrift://hive-metastore.database.svc.cluster.local:9083")
        .enableHiveSupport()
        .config("spark.hadoop.fs.s3a.endpoint", f"http://{MINIO_ENDPOINT}")
        .config("spark.hadoop.fs.s3a.access.key", MINIO_ACCESS_KEY)
        .config("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET_KEY)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .getOrCreate()
    )

    bronze_bureau_balance_table = spark.read.table("homecredit_bronze.merged_bureau")

    silver_bureau_balance_table = bronze_bureau_balance_table.select(
        "sk_id_bureau",
        "sk_id_curr",
        "months_balance",
        "status",
        "updated",
        "event_ts"
    )

    bureau_table = (
        spark.read.format("csv")
        .option("header", "true")
        .load("s3a://sample-data/bureau.csv")
    )

    for col in bureau_table.columns:
        bureau_table = bureau_table.withColumnRenamed(col, col.lower())
    
    silver_bureau_table_scd2 = add_scd_type2_cols(bureau_table, ts_col=None) \
        .withColumn("event_ts", F.current_timestamp())
        
    spark.sql("CREATE SCHEMA IF NOT EXISTS homecredit_silver")

    write_with_delta_sql(
        spark,
        silver_bureau_table_scd2,
        table_name="dim_bureau",
        schema="homecredit_silver",      
        base_path=SILVER_PATH_MERGED,
        table_ddl=silver_dim_bureau_schema,
        mode="append"
    )

    write_with_delta_sql(
        spark,
        silver_bureau_balance_table,
        table_name="fact_bureau_balance",
        schema="homecredit_silver",
        base_path=SILVER_PATH_MERGED,
        table_ddl=silver_fact_bureau_balance_schema,
        mode="append"
    )


    
if __name__ == "__main__":
    main()