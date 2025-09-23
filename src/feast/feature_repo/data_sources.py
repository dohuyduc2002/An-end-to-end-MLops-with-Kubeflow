from feast import KafkaSource

from feast.infra.offline_stores.contrib.spark_offline_store.spark_source import (
    SparkSource,
)
from feast.data_format import JsonFormat

import os
from minio import Minio

s3_access_key = os.getenv("S3_ACCESS_KEY")
s3_secret_key = os.getenv("S3_SECRET_KEY")

minio_client = Minio(
    "minio-server.example.com",
    access_key=s3_access_key,
    secret_key=s3_secret_key,
    secure=False,
)


def create_spark_source(spark_src_name, minio_path):
    batching_source = SparkSource(
        name=spark_src_name,
        path=minio_path,
        file_format="delta",
        timestamp_field="updated",
        created_timestamp_column="created",
        description="user spark source",
        owner="dohuyduc.work@gmail.com",
    )
    return batching_source


def create_kafka_source(kafka_src_name, topic, kafka_bootstrap_servers, stream_schema, watermark_delay_threshold=None):
    # A push source is useful if you have upstream systems that transform features (e.g. stream processing jobs)
    stream_source = KafkaSource(
        name=kafka_src_name,
        timestamp_field="updated",
        message_format=JsonFormat(stream_schema),
        kafka_bootstrap_servers=kafka_bootstrap_servers,
        topic=topic,
        watermark_delay_threshold= watermark_delay_threshold,
        description="user kafka source",
        owner="dohuyduc.work@gmail.com"
    )
    return stream_source