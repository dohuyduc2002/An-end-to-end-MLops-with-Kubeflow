from pyflink.common.typeinfo import Types

from confluent_kafka.schema_registry import SchemaRegistryClient, Schema
from confluent_kafka.schema_registry.avro import AvroDeserializer, AvroSerializer
from confluent_kafka.serialization import MessageField, SerializationContext
from pyflink.common.serialization import DeserializationSchema

SCHEMA_REGISTRY_CONF = {
    "url": "http://schema-registry-svc.infrastructure.svc.cluster.local:8081"
}
BOOTSTRAP = "kafka-cluster-0-kafka-bootstrap.kafka.svc.cluster.local:9092"
GROUP_ID = "flink-bureau-merge"
INPUT_BUREAU = "bureau"
INPUT_BAL = "bureau_balance"
OUTPUT_TOPIC = "merged_bureau"

schema_registry_client = SchemaRegistryClient(SCHEMA_REGISTRY_CONF)


class AvroSerializerWrapper:
    def __init__(self, schema_registry_conf, topic):
        self.schema_registry_conf = schema_registry_conf
        self.topic = topic
        self.subject_name = f"{topic}-value"
        self.schema_registry_client, self._serializer, self.schema_str, self.ctx = (
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

    def get_serializer(self):
        if self._serializer is None:
            self.ctx = SerializationContext(self.topic, MessageField.VALUE)
            self.schema_registry_client = SchemaRegistryClient(
                self.schema_registry_conf
            )
            self.get_schema()
            self._serializer = AvroSerializer(
                self.schema_registry_client, self.schema_str
            )
        return self._serializer

    def serialize(self, record):
        try:
            return self.get_serializer()(record, self.ctx)
        except:
            print("[INFO] Invalid record !!!")
            return None


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
        try:
            return self.get_deserializer()(record, self.ctx)
        except:
            print("[INFO] Invalid record !!!")
            return None


class AvroDeserializationSchema(DeserializationSchema):
    def __init__(self, topic):
        self.deserializer = AvroDeserializerWrapper(SCHEMA_REGISTRY_CONF, topic)

    def deserialize(self, message: bytes):
        return self.deserializer.deserialize(message)  # dict

    def is_end_of_stream(self, next_element):
        return False

    def get_produced_type(self):
        return Types.PICKLED_BYTE_ARRAY()


def register_schema(subject_name, schema_path):
    subjects = schema_registry_client.get_subjects()
    if subject_name in subjects:
        print(f"[INFO] Schema {subject_name} already existed !!!")
    else:
        with open(schema_path, "r") as f:
            schema_str = f.read()
        schema = Schema(schema_str, schema_type="AVRO")
        schema_id = schema_registry_client.register_schema(subject_name, schema)
        print(f"[INFO] Register Schema {subject_name} successfully !!!")


def get_schema(subject_name):
    subjects = schema_registry_client.get_subjects()
    assert subject_name in subjects
    schema_str = schema_registry_client.get_latest_version(
        subject_name
    ).schema.schema_str
    return schema_str
