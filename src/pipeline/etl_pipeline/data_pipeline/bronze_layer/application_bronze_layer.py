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
    MINIO_ENDPOINT,
    MINIO_ACCESS_KEY,
    MINIO_SECRET_KEY,
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
        .appName("Bronze application batching")
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
        CREATE TABLE IF NOT EXISTS homecredit_bronze.application (
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

        (
            stg_upsert.write
            .format("delta")
            .mode("append")
            .save(BRONZE_PATH_APPLICATION)
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
