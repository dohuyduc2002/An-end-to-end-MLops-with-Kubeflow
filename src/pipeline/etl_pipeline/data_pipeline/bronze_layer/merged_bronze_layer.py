from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, to_timestamp, to_date
from pyspark.sql import functions as F

import sys
from pathlib import Path
from bronze_etl_schema import bronze_merged_schema
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    KAFKA_BOOTSTRAP_SERVERS,
    MERGED_TOPIC,
    BRONZE_PATH_MERGED,
    CHECKPOINT_PATH_MERGED,
    MINIO_ENDPOINT,
    MINIO_ACCESS_KEY,
    MINIO_SECRET_KEY,
)


def main():
    spark = (
        SparkSession.builder
        .appName("Bronze Flink bureau batching")
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

    spark.conf.set("spark.sql.session.timeZone", "UTC")

    spark.sql("CREATE SCHEMA IF NOT EXISTS homecredit_bronze")
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS homecredit_bronze.merged_bureau (
            {bronze_merged_schema.toDDL()}
        )
        USING delta
        LOCATION '{BRONZE_PATH_MERGED}'
    """)


    # Kafka source
    kafka_df = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", MERGED_TOPIC)
        .option("startingOffsets", "earliest")
        .load()
    )

    kafka_df_standardized = (
        kafka_df.selectExpr("CAST(value AS STRING) AS value_str")
        .select(from_json(col("value_str"), bronze_merged_schema).alias("kafka_data"))
        .select(
            col("kafka_data.sk_id_bureau").alias("sk_id_bureau"),
            col("kafka_data.sk_id_curr").alias("sk_id_curr"),
            col("kafka_data.months_balance").alias("months_balance"),
            col("kafka_data.status").alias("status"),
            to_timestamp(col("kafka_data.updated")).alias("event_ts"),
        )
        .withColumn("dt", to_date(col("event_ts")))
    )

    def process(batch_df, batch_id):
        (
            batch_df.withColumn("batch_id", F.lit(batch_id))
            .write.format("delta")
            .mode("append")
            .save(BRONZE_PATH_MERGED)
        )

    kafka_df_standardized.writeStream \
        .option("checkpointLocation", CHECKPOINT_PATH_MERGED) \
        .option("maxOffsetsPerTrigger", 100000) \
        .trigger(processingTime="60 seconds") \
        .foreachBatch(process) \
        .start() \
        .awaitTermination()


if __name__ == "__main__":
    main()
