from pyspark.sql import SparkSession
from pyspark.sql import functions as F

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    MINIO_ENDPOINT,
    MINIO_ACCESS_KEY,
    MINIO_SECRET_KEY,
    GOLD_PATH,
    SILVER_PATH_MERGED,
)


def main():
    def write_delta_overwrite(df, table_name):
        path = f"s3a://{GOLD_PATH}/{table_name}"
        (
            df.write.format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .save(path)
        )
        print(f"[OK] Wrote Delta -> {path}")

    spark = (
        SparkSession.builder.master("local[*]")
        .config("spark.hadoop.fs.s3a.endpoint", f"http://{MINIO_ENDPOINT}")
        .config("spark.hadoop.fs.s3a.access.key", MINIO_ACCESS_KEY)
        .config("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET_KEY)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.databricks.delta.schema.autoMerge.enabled", "true")
        .appName("Gold merge batching")
        .getOrCreate()
    )

    silver_bureau_balance_table = spark.read.format("delta").load(SILVER_PATH_MERGED)

    bureau_table = (
        spark.read.format("csv")
        .option("header", "true")
        .load("s3a://sample-data/bureau.csv")
    )

    for c in bureau_table.columns:
        bureau_table = bureau_table.withColumnRenamed(c, c.lower())
        
    write_delta_overwrite(bureau_table, "dim_bureau")
    write_delta_overwrite(silver_bureau_balance_table, "fact_bureau_balance")
    
