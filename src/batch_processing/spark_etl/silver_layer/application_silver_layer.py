from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroDeserializer
from confluent_kafka.serialization import MessageField, SerializationContext
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

import json
import sys
from pathlib import Path

from spark_schema import cdc_schema, silver_schema

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    MINIO_ENDPOINT,
    MINIO_ACCESS_KEY,
    MINIO_SECRET_KEY,
    KAFKA_BOOTSTRAP_SERVERS,
    APPLICATION_TOPIC,
    SCHEMA_REGISTRY_CONFIG,
    SILVER_PATH_APPLICATION,
    CHECKPOINT_PATH_APPLICATION,
)

class AvroDeserializerWrapper:
    def __init__(self, schema_registry_conf, topic):
        self.schema_registry_conf = schema_registry_conf
        self.topic = topic
        self.subject_name = f"{topic}-value"
        self.schema_registry_client, self._deserializer, self.schema_str, self.ctx = (
            None,
            None,
            None,
            None,
        )

    def get_schema(self):
        assert self.schema_registry_client
        subjects = self.schema_registry_client.get_subjects()
        assert self.subject_name in subjects
        self.schema_str = self.schema_registry_client.get_latest_version(
            self.subject_name
        ).schema.schema_str

    def get_deserializer(self):
        if self._deserializer is None:
            self.ctx = SerializationContext(self.topic, MessageField.VALUE)
            self.schema_registry_client = SchemaRegistryClient(
                self.schema_registry_conf
            )
            self.get_schema()
            self._deserializer = AvroDeserializer(
                self.schema_registry_client, self.schema_str
            )
        return self._deserializer

    def deserialize(self, record):
        raw_dict = self.get_deserializer()(record, self.ctx)
        return raw_dict


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
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .appName("Explode CDC batching")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("DEBUG")
    spark.conf.set("spark.sql.session.timeZone", "UTC")

    kafka_df = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", TOPIC)
        .option("startingOffsets", "earliest")
        .load()
    )

    deserializer = AvroDeserializerWrapper(schema_registry_conf=SCHEMA_REGISTRY_CONFIG, topic=TOPIC)

    @udf(returnType=StringType())
    def value_to_json(batch):
        return json.dumps(deserializer.deserialize(batch)) if batch is not None else None

    def process(batch_df, batch_id):
        parsed = (
            batch_df.select(
                value_to_json(F.col("value")).alias("val_json"), "timestamp"
            )
            .withColumn("env", F.from_json("val_json", cdc_schema))
            .select(
                "env.*", F.col("timestamp").alias("ingest_ts")
            )
            .withColumn(
                "event_ts",
                F.to_timestamp((F.col("ts_ms") / F.lit(1000)).cast("double")),
            )
        )

        stg_upsert = (
            parsed.filter(F.col("op") != F.lit("d"))
            .withColumn("obj", F.from_json(F.col("after"), silver_schema))
            .select("obj.*", "event_ts")
        )

        spark_col = silver_schema.toDDL()

        spark.sql(
            f"""
        CREATE TABLE IF NOT EXISTS delta.`{SILVER_PATH_APPLICATION}` (
        {spark_col},
        event_ts TIMESTAMP
        )
        USING delta
        """
        )

        stg_upsert.write.mode("append").format("delta").save(SILVER_PATH_APPLICATION)

    kafka_df.writeStream.option("checkpointLocation", CHECKPOINT_PATH_APPLICATION).trigger(processingTime="30 seconds") \
    .foreachBatch(process) \
    .start() \
    .awaitTermination()

if __name__ == "__main__":
    main()
