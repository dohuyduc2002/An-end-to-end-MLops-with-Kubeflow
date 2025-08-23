from typing import Any, Dict, Optional
from datetime import datetime, timezone

from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.functions import CoProcessFunction, RuntimeContext
from pyflink.datastream.state import ValueStateDescriptor, ListStateDescriptor
from pyflink.common.typeinfo import Types
from pyflink.common import Row


from pyflink.table import (
    StreamTableEnvironment,
    EnvironmentSettings,
    Schema,
    DataTypes,
)

from utils import (debezium_ms_to_iso_utc, kafka_bureau_balance_table, 
                   kafka_bureau_table, flink_flatten_bureau_balance_table, 
                   flink_flatten_bureau_table, output_table)
import os

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP")
SCHEMA_REGISTRY_URL = os.getenv("SCHEMA_REGISTRY_URL")
INPUT_BUREAU_TOPIC = os.getenv("INPUT_BUREAU_TOPIC")
INPUT_BAL_TOPIC = os.getenv("INPUT_BALANCE_TOPIC")
OUTPUT_TOPIC = os.getenv("OUTPUT_TOPIC")

# Convert Row from Table API to dict for easier processing in DataStream API
def map_bureau_row(row) -> Dict[str, Any]:
    return {
        "sk_id_bureau": row[0],
        "sk_id_curr": row[1],
        "credit_active": row[2],
        "credit_currency": row[3],
        "days_credit": row[4],
        "credit_day_overdue": row[5],
        "days_credit_enddate": row[6],
        "days_enddate_fact": row[7],
        "amt_credit_max_overdue": row[8],
        "cnt_credit_prolong": row[9],
        "amt_credit_sum": row[10],
        "amt_credit_sum_debt": row[11],
        "amt_credit_sum_limit": row[12],
        "amt_credit_sum_overdue": row[13],
        "credit_type": row[14],
        "days_credit_update": row[15],
        "amt_annuity": row[16],
        "ts_ms": row[17],  
        "event_time": row[18],  
    }

def map_bal_row(row) -> Dict[str, Any]:
    return {
        "sk_id_bureau": row[0],
        "months_balance": row[1],
        "status": row[2],
        "ts_ms": row[3],
        "event_time": row[4],
    }

# =============================
# MERGE (DataStream)
# =============================
class MergeBureauWithBalance(CoProcessFunction):
    def open(self, ctx: RuntimeContext):
        self.bureau = ctx.get_state(
            ValueStateDescriptor("bureau", Types.PICKLED_BYTE_ARRAY())
        )
        # In this case, bureau_balance is a ListState to hold multiple balances
        self.bureau_balance = ctx.get_list_state(
            ListStateDescriptor("bureau_balance", Types.PICKLED_BYTE_ARRAY())
        )

    def process_element1(self, bureau_dict: Dict[str, Any], ctx: CoProcessFunction.Context):
        if bureau_dict is None:
            return
        self.bureau.update(bureau_dict)

        for bal in list(self.bureau_balance.get()):
            # Due to balance being a list state, we need to iterate over it and flush to stream
            out = self._merge(bureau_dict, bal)
            yield out
        self.bureau_balance.clear()  # clear processed balances

    def process_element2(self, balance_dict: Dict[str, Any], ctx: CoProcessFunction.Context):
        if balance_dict is None:
            return
        bureau = self.bureau.value()
        if bureau:
            yield self._merge(bureau, balance_dict)
        else:
            # If there is no bureau state, add the balance to the waiting list
            self.bureau_balance.add(balance_dict)

    def _merge(self, bureau_dict, balance_dict):
        out = dict(bureau_dict)
        out["sk_id_bureau"] = balance_dict.get("sk_id_bureau", out.get("sk_id_bureau"))
        out["months_balance"] = balance_dict.get("months_balance")
        out["status"] = balance_dict.get("status")
        timestamp_1 = bureau_dict.get("ts_ms")
        timestamp_2 = balance_dict.get("ts_ms")
        ts_max = max([t for t in [timestamp_1, timestamp_2] if t is not None], default=None)
        out["updated"] = debezium_ms_to_iso_utc(ts_max)
        return out


# =============================
# MAIN
# =============================
def main():
    # ---- Envs ----
    env = StreamExecutionEnvironment.get_execution_environment()
    settings = EnvironmentSettings.in_streaming_mode()
    t_env = StreamTableEnvironment.create(env, environment_settings=settings)

    # ---- DDL from kafka topic ----
    # Bureau
    t_env.execute_sql(kafka_bureau_table)
    t_env.execute_sql(kafka_bureau_balance_table)
    
    # ---- Create view
    bureau_after = t_env.sql_query(flink_flatten_bureau_table)
    balance_after = t_env.sql_query(flink_flatten_bureau_balance_table)

    # ---- Table -> DataStream (Row -> dict) ----
    ds_bureau = t_env.to_data_stream(bureau_after).map(map_bureau_row)
    ds_balance = t_env.to_data_stream(balance_after).map(map_bal_row)

    merged = (
        ds_bureau.key_by(lambda d: d["sk_id_bureau"])
        .connect(ds_balance.key_by(lambda d: d["sk_id_bureau"]))
        .process(MergeBureauWithBalance(), output_type=Types.PICKLED_BYTE_ARRAY())
    )

    # ---- DataStream -> Table ----
    out_schema = (
        Schema.new_builder()
        .column("sk_id_bureau", DataTypes.BIGINT())
        .column("sk_id_curr", DataTypes.BIGINT())
        .column("months_balance", DataTypes.INT())
        .column("status", DataTypes.STRING())
        .column("updated", DataTypes.STRING())
        .build()
    )

    def to_row(data_dict: Dict[str, Any]) -> Row:
        return Row(
            sk_id_bureau=data_dict.get("sk_id_bureau"),
            sk_id_curr=data_dict.get("sk_id_curr"),
            months_balance=data_dict.get("months_balance"),
            status=data_dict.get("status"),
            updated=data_dict.get("updated"),
        )

    merged_rows = merged.map(
        to_row,
        output_type=Types.ROW_NAMED(
            ["sk_id_bureau", "sk_id_curr", "months_balance", "status", "updated"],
            [Types.LONG(), Types.LONG(), Types.INT(), Types.STRING(), Types.STRING()],
        ),
    )

    # Convert DataStream to Table
    out_table = t_env.from_data_stream(merged_rows, out_schema)

    #Sink to kafka in json format
    t_env.execute_sql(output_table)

    t_env.create_temporary_view("merged_stream", out_table)
    t_env.execute_sql("INSERT INTO merged_out SELECT * FROM merged_stream")


if __name__ == "__main__":
    main()
