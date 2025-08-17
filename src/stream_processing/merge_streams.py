from typing import Any, Dict, Optional
from datetime import datetime, timezone

from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors.kafka import (
    KafkaSource,
    KafkaSink,
    KafkaRecordSerializationSchema,
    KafkaOffsetsInitializer,
    DeliveryGuarantee,
)
from pyflink.datastream.functions import CoProcessFunction, RuntimeContext
from pyflink.common.typeinfo import Types
from pyflink.datastream.state import ValueStateDescriptor, ListStateDescriptor
from pyflink.common.serialization import DeserializationSchema, SerializationSchema

# ---- utils bạn đã có ----
from utils import (
    AvroSerializerWrapper,
    AvroDeserializerWrapper,
    register_schema,
    SCHEMA_REGISTRY_CONF,
    BOOTSTRAP,
    GROUP_ID,
    INPUT_BUREAU,
    INPUT_BAL,
    OUTPUT_TOPIC,
)


# =============================
# Helpers
# =============================
def _ms_to_iso_utc(ms: Optional[int]) -> Optional[str]:
    if ms is None:
        return None
    return (
        datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _enveloped_to_flat_with_updated(env: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Debezium envelope -> flat dict (lower-case keys) + 'updated' (ISO-8601 UTC).
    Bỏ qua delete events (op='d').
    """
    if not env or env.get("op") == "d":
        return None
    after_col = env.get("after")
    if not after_col:
        return None

    out = {(k.lower() if isinstance(k, str) else k): v for k, v in after_col.items()}
    out["updated"] = _ms_to_iso_utc(env.get("ts_ms"))
    return out


def _extract_key_from_env(env_obj: Dict[str, Any]) -> str:
    """
    KeyBy theo sk_id_bureau từ envelope (chưa flat).
    Điều chỉnh nếu upstream đổi tên cột.
    """
    try:
        after = env_obj.get("after") or {}
        key_val = after.get("SK_ID_BUREAU") or after.get("sk_id_bureau")
        return str(key_val) if key_val is not None else ""
    except Exception:
        return ""


# =============================
# Deserializer cho SOURCE (Avro -> dict)
# =============================
class AvroDictDeserializationSchema(DeserializationSchema):
    """
    Deserialize value Avro (Schema Registry) -> Python dict (envelope).
    """

    def __init__(self, topic: str):
        self.topic = topic
        self._des = AvroDeserializerWrapper(SCHEMA_REGISTRY_CONF, topic)

    def deserialize(self, message: bytes) -> Optional[Dict[str, Any]]:
        try:
            return self._des.deserialize(message)
        except Exception as e:
            print(f"[WARN] Failed to deserialize from topic={self.topic}: {e}")
            return None

    def is_end_of_stream(self, next_element) -> bool:
        return False

    def get_produced_type(self):
        return Types.PICKLED_BYTE_ARRAY()


# =============================
# Serializer cho SINK
# =============================
class KeyFromDictSerializationSchema(SerializationSchema):
    """
    Key = str(sk_id_bureau), encode utf-8.
    """

    def serialize(self, element: Dict[str, Any]) -> bytes:
        key = str(element.get("sk_id_bureau", ""))
        return key.encode("utf-8")


class AvroValueSerializationSchema(SerializationSchema):
    """
    Value = dict -> Avro bytes (Schema Registry).
    Subject là <topic>-value (do AvroSerializerWrapper).
    """

    def __init__(self, topic: str):
        self._ser = AvroSerializerWrapper(SCHEMA_REGISTRY_CONF, topic)

    def serialize(self, element: Dict[str, Any]) -> bytes:
        try:
            b = self._ser.serialize(element)
            return b or b""
        except Exception as e:
            print(f"[WARN] Failed to serialize to Avro: {e}")
            return b""


# =============================
# CoProcessFunction merge logic
# =============================
class MergeBureauWithBalance(CoProcessFunction):
    """
    Cả 2 stream đã keyBy(sk_id_bureau) trước khi connect.
    State per key:
      - bureau_state: record bureau mới nhất (dict, đã flat + 'updated')
      - bal_state: list record balance chờ merge (đã flat + 'updated')
    Khi có đủ, emit merged:
        {bureau_fields..., months_balance, status, updated}
    'updated' ưu tiên theo balance nếu có; nếu không có thì dùng từ bureau.
    """

    def open(self, ctx: RuntimeContext):
        self.bureau_state = ctx.get_state(
            ValueStateDescriptor("bureau", Types.PICKLED_BYTE_ARRAY())
        )
        self.bal_state = ctx.get_list_state(
            ListStateDescriptor("bureau_balance", Types.PICKLED_BYTE_ARRAY())
        )

    # element1 = bureau (envelope)
    def process_element1(self, value: Dict[str, Any], ctx: CoProcessFunction.Context):
        if value is None:
            return
        flat_bureau = _enveloped_to_flat_with_updated(value)
        if flat_bureau is None:
            return

        # lưu bureau mới nhất
        self.bureau_state.update(flat_bureau)

        # nếu có balance chờ -> flush
        balances = list(self.bal_state.get())
        if balances:
            for bal in balances:
                merged = dict(flat_bureau)
                merged["sk_id_bureau"] = bal.get(
                    "sk_id_bureau", merged.get("sk_id_bureau")
                )
                merged["months_balance"] = bal.get("months_balance")
                merged["status"] = bal.get("status")
                merged["updated"] = bal.get("updated") or merged.get("updated")
                ctx.output(None, merged)
            self.bal_state.clear()

    # element2 = bureau_balance (envelope)
    def process_element2(self, value: Dict[str, Any], ctx: CoProcessFunction.Context):
        if value is None:
            return
        flat_bal = _enveloped_to_flat_with_updated(value)
        if flat_bal is None:
            return

        bureau = self.bureau_state.value()
        if bureau:
            merged = dict(bureau)
            merged["sk_id_bureau"] = flat_bal.get(
                "sk_id_bureau", merged.get("sk_id_bureau")
            )
            merged["months_balance"] = flat_bal.get("months_balance")
            merged["status"] = flat_bal.get("status")
            merged["updated"] = flat_bal.get("updated") or merged.get("updated")
            ctx.output(None, merged)
        else:
            # chưa có bureau -> lưu tạm
            self.bal_state.add(flat_bal)


# =============================
# Main
# =============================
def main():
    # ---- Flink env ----
    env = StreamExecutionEnvironment.get_execution_environment()
    # Bật checkpoint để bảo đảm (ít nhất) AT_LEAST_ONCE
    env.enable_checkpointing(60_000)  # 60s, chỉnh theo nhu cầu

    # ---- Kafka config ----

    # (tuỳ chọn) Đăng ký schema output nếu chưa có
    # register_schema(f"{OUTPUT_TOPIC}-value", "/app/schemas/merged_bureau.avsc")

    # ---- Sources (Avro -> dict envelope) ----
    bureau_source = (
        KafkaSource.builder()
        .set_bootstrap_servers(BOOTSTRAP)
        .set_group_id(GROUP_ID)
        .set_topics(INPUT_BUREAU)
        .set_value_only_deserializer(AvroDictDeserializationSchema(INPUT_BUREAU))
        .set_starting_offsets(KafkaOffsetsInitializer.earliest())
        .build()
    )

    bal_source = (
        KafkaSource.builder()
        .set_bootstrap_servers(BOOTSTRAP)
        .set_group_id(GROUP_ID)
        .set_topics(INPUT_BAL)
        .set_value_only_deserializer(AvroDictDeserializationSchema(INPUT_BAL))
        .set_starting_offsets(KafkaOffsetsInitializer.earliest())
        .build()
    )

    bureau_stream = env.from_source(
        bureau_source, watermark_strategy=None, source_name="bureau"
    )
    bal_stream = env.from_source(
        bal_source, watermark_strategy=None, source_name="bureau_balance"
    )

    # KeyBy trước khi connect để có per-key state trong CoProcess
    keyed_bureau = bureau_stream.key_by(_extract_key_from_env)
    keyed_bal = bal_stream.key_by(_extract_key_from_env)

    merged = keyed_bureau.connect(keyed_bal).process(MergeBureauWithBalance())

    # ---- Sink: key=str, value=Avro ----
    output_sink = (
        KafkaSink.builder()
        .set_bootstrap_servers(BOOTSTRAP)
        .set_record_serializer(
            KafkaRecordSerializationSchema.builder()
            .set_topic(OUTPUT_TOPIC)
            .set_key_serialization_schema(KeyFromDictSerializationSchema())
            .set_value_serialization_schema(AvroValueSerializationSchema(OUTPUT_TOPIC))
            .build()
        )
        .set_delivery_guarantee(DeliveryGuarantee.AT_LEAST_ONCE)
        # EXACTLY_ONCE (tuỳ chọn):
        # .set_delivery_guarantee(DeliveryGuarantee.EXACTLY_ONCE)
        # .set_transactional_id_prefix("flink-merged-bureau-")  # unique per job/cluster
        .build()
    )

    merged.sink_to(output_sink)

    env.execute("merge_bureau_to_json")


if __name__ == "__main__":
    register_schema(
        "merged_bureau-value", "/opt/flink/usrlib/schemas/merged_bureau.avsc"
    )
    main()
