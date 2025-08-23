from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, to_timestamp, to_date
from pyspark.sql.types import StructType, StructField, LongType, IntegerType, StringType

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, KAFKA_BOOTSTRAP_SERVERS, MERGED_TOPIC, SILVER_PATH_MERGED, CHECKPOINT_PATH_MERGED

def main():
    spark = (
        SparkSession.builder.master("local[*]")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.hadoop.fs.s3a.endpoint", f"http://{MINIO_ENDPOINT}")
        .config("spark.hadoop.fs.s3a.access.key", MINIO_ACCESS_KEY)
        .config("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET_KEY)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config(
            "spark.hadoop.fs.s3a.impl",
            "org.apache.hadoop.fs.s3a.S3AFileSystem",
        )
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .appName("Flink batching")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("DEBUG")
    spark.conf.set("spark.sql.session.timeZone", "UTC")

    merged_schema = StructType(
        [
            StructField("sk_id_bureau", LongType(), True),
            StructField("sk_id_curr", LongType(), True),
            StructField("months_balance", IntegerType(), True),
            StructField("status", StringType(), True),
            StructField("updated", StringType(), True),
        ]
    )

    kafka_df = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", MERGED_TOPIC)
        .option("startingOffsets", "earliest")
        .load()
    )

    kafka_df_standardized = (
        kafka_df.selectExpr("CAST(value AS STRING) AS value_str")
        .select(from_json(col("value_str"), merged_schema).alias("kafka_data"))
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
        (batch_df.write
            .format("delta")
            .mode("append")
            .option("mergeSchema", "true")
            .save(SILVER_PATH_MERGED)
        )

        spark.sql(f"""
            CREATE OR REPLACE TEMP VIEW silver_layer AS
            SELECT * FROM delta.`{SILVER_PATH_MERGED}`
        """)

    kafka_df_standardized.writeStream \
        .option("checkpointLocation", CHECKPOINT_PATH_MERGED) \
        .trigger(processingTime="30 seconds") \
        .foreachBatch(process) \
        .start() \
        .awaitTermination()

if __name__ == "__main__":
    main()
