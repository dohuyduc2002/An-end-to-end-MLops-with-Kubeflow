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

from bronze_etl_schema import cdc_schema, bronze_application_schema
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    KAFKA_BOOTSTRAP_SERVERS,
    APPLICATION_TOPIC,
    SCHEMA_REGISTRY_CONFIG,
    BRONZE_PATH_APPLICATION,
    CHECKPOINT_PATH_APPLICATION,
)


class AvroDeserializerWrapper:
    def __init__(self, schema_registry_conf, topic):
        self.schema_registry_conf = schema_registry_conf
        self.topic = topic
        self.subject_name = f"{topic}-value"
        self.schema_registry_client, self._deserializer, self.schema_str, self.ctx = (
            None, None, None, None
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
            self.schema_registry_client = SchemaRegistryClient(self.schema_registry_conf)
            self.get_schema()
            self._deserializer = AvroDeserializer(
                self.schema_registry_client, self.schema_str
            )
        return self._deserializer

    def deserialize(self, record):
        return self.get_deserializer()(record, self.ctx)


def main():
    spark = (
        SparkSession.builder
        .config("spark.hadoop.fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem")
        .config("spark.hadoop.fs.AbstractFileSystem.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS")
        .config("spark.hadoop.google.cloud.auth.service.account.enable", "true")
        .config("spark.hadoop.google.cloud.auth.service.account.json.keyfile", "/var/secrets/gcp/gcp-key.json")
        .appName("Explode CDC batching")
        .getOrCreate()
    )
    spark.conf.set("spark.sql.session.timeZone", "UTC")

    # Ensure schema + UC table
    spark.sql("DROP TABLE IF EXISTS bronze.application")
    spark.sql(f"""
        CREATE SCHEMA IF NOT EXISTS bronze
        LOCATION 'gs://unity-catalog-dhduc/bronze'
    """)
    spark.sql(f"""
        CREATE TABLE bronze.application (
            {bronze_application_schema.toDDL()}
        )
        USING delta
        LOCATION '{BRONZE_PATH_APPLICATION}'
    """)

    # Kafka source
    kafka_df = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", APPLICATION_TOPIC)
        .option("startingOffsets", "earliest")
        .load()
    )

    deserializer = AvroDeserializerWrapper(
        schema_registry_conf=SCHEMA_REGISTRY_CONFIG, topic=APPLICATION_TOPIC
    )

    @udf(returnType=StringType())
    def value_to_json(batch):
        return json.dumps(deserializer.deserialize(batch)) if batch else None

    def process(batch_df, batch_id):
        parsed = (
            batch_df.select(value_to_json(F.col("value")).alias("val_json"), "timestamp")
            .withColumn("env", F.from_json("val_json", cdc_schema))
            .select("env.*", F.col("timestamp").alias("ingest_ts"))
            .withColumn(
                "event_ts", F.to_timestamp((F.col("ts_ms") / F.lit(1000)).cast("double"))
            )
        )

        stg_upsert = (
            parsed.filter(F.col("op") != F.lit("d"))
            .withColumn("obj", F.from_json(F.col("after"), bronze_application_schema))
            .select("obj.*")
            .withColumn("batch_id", F.lit(batch_id))
        )

        # Ghi thẳng vào UC table
        (
            stg_upsert.write
            .format("delta")                                  # định dạng Delta
            .mode("append")                                   # hoặc overwrite
            .option("path", BRONZE_PATH_APPLICATION)          # path GCS/abfss/S3
            .saveAsTable("bronze.application")                # UC table name
        )



    kafka_df.writeStream \
        .option("checkpointLocation", CHECKPOINT_PATH_APPLICATION) \
        .option("maxOffsetsPerTrigger", 100000) \
        .trigger(processingTime="60 seconds") \
        .foreachBatch(process) \
        .start() \
        .awaitTermination()


if __name__ == "__main__":
    main()
