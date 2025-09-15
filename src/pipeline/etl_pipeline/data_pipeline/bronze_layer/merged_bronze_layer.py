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
)


def main():
    spark = (
        SparkSession.builder
        .config("spark.hadoop.fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem")
        .config("spark.hadoop.fs.AbstractFileSystem.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS")
        .config("spark.hadoop.google.cloud.auth.service.account.enable", "true")
        .config("spark.hadoop.google.cloud.auth.service.account.json.keyfile", "/var/secrets/gcp/gcp-key.json")
        .appName("Flink batching")
        .getOrCreate()
    )
    spark.conf.set("spark.sql.session.timeZone", "UTC")

    # Ensure schema + UC table
    spark.sql("DROP TABLE IF EXISTS bronze.merged_bureau")
    spark.sql(f"""
        CREATE SCHEMA IF NOT EXISTS bronze
        LOCATION 'gs://unity-catalog-dhduc/bronze'
    """)
    spark.sql(f"""
        CREATE TABLE bronze.merged_bureau (
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
            .write
            .format("delta")
            .mode("append")
            .option("path", BRONZE_PATH_MERGED)  # hoặc biến BRONZE_PATH_MERGED_BUREAU
            .saveAsTable("bronze.merged_bureau")
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
