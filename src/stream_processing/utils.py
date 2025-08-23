from datetime import datetime, timezone
import os

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP")
SCHEMA_REGISTRY_URL = os.getenv("SCHEMA_REGISTRY_URL")
INPUT_BUREAU_TOPIC = os.getenv("INPUT_BUREAU_TOPIC")
INPUT_BAL_TOPIC = os.getenv("INPUT_BALANCE_TOPIC")
OUTPUT_TOPIC = os.getenv("OUTPUT_TOPIC")


def debezium_ms_to_iso_utc(ts_ms):
    dt_utc = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).replace(tzinfo=None)
    return dt_utc


kafka_bureau_table = f"""
    CREATE TABLE bureau_raw (
      `before` ROW<
        sk_id_curr BIGINT,
        sk_id_bureau BIGINT,
        credit_active STRING,
        credit_currency STRING,
        days_credit INT,
        credit_day_overdue INT,
        days_credit_enddate DOUBLE,
        days_enddate_fact DOUBLE,
        amt_credit_max_overdue DOUBLE,
        cnt_credit_prolong INT,
        amt_credit_sum DOUBLE,
        amt_credit_sum_debt DOUBLE,
        amt_credit_sum_limit DOUBLE,
        amt_credit_sum_overdue DOUBLE,
        credit_type STRING,
        days_credit_update INT,
        amt_annuity DOUBLE
      >,
      `after` ROW<
        sk_id_curr BIGINT,
        sk_id_bureau BIGINT,
        credit_active STRING,
        credit_currency STRING,
        days_credit INT,
        credit_day_overdue INT,
        days_credit_enddate DOUBLE,
        days_enddate_fact DOUBLE,
        amt_credit_max_overdue DOUBLE,
        cnt_credit_prolong INT,
        amt_credit_sum DOUBLE,
        amt_credit_sum_debt DOUBLE,
        amt_credit_sum_limit DOUBLE,
        amt_credit_sum_overdue DOUBLE,
        credit_type STRING,
        days_credit_update INT,
        amt_annuity DOUBLE
      >,
      op STRING,
      ts_ms BIGINT,
      event_time AS TO_TIMESTAMP_LTZ(ts_ms, 3),
      WATERMARK FOR event_time AS event_time - INTERVAL '5' SECOND
    ) WITH (
      'connector' = 'kafka',
      'topic' = '{INPUT_BUREAU_TOPIC}',
      'properties.bootstrap.servers' = '{BOOTSTRAP}',
      'properties.group.id' = 'g-bureau',
      'scan.startup.mode' = 'earliest-offset',
      'value.format' = 'avro-confluent',
      'value.avro-confluent.url' = '{SCHEMA_REGISTRY_URL}'
    )
    """

kafka_bureau_balance_table = f"""
    CREATE TABLE bureau_balance_raw (
      `before` ROW<
        sk_id_bureau BIGINT,
        months_balance INT,
        status STRING
      >,
      `after` ROW<
        sk_id_bureau BIGINT,
        months_balance INT,
        status STRING
      >,
      op STRING,
      ts_ms BIGINT,
      event_time AS TO_TIMESTAMP_LTZ(ts_ms, 3),
      WATERMARK FOR event_time AS event_time - INTERVAL '5' SECOND
    ) WITH (
      'connector' = 'kafka',
      'topic' = '{INPUT_BAL_TOPIC}',
      'properties.bootstrap.servers' = '{BOOTSTRAP}',
      'properties.group.id' = 'g-bureau-bal',
      'scan.startup.mode' = 'earliest-offset',
      'value.format' = 'avro-confluent',
      'value.avro-confluent.url' = '{SCHEMA_REGISTRY_URL}'
    )
    """

flink_flatten_bureau_table = """
      SELECT
        (after).sk_id_bureau AS sk_id_bureau,
        (after).sk_id_curr AS sk_id_curr,
        (after).credit_active AS credit_active,
        (after).credit_currency AS credit_currency,
        (after).days_credit AS days_credit,
        (after).credit_day_overdue AS credit_day_overdue,
        (after).days_credit_enddate AS days_credit_enddate,
        (after).days_enddate_fact AS days_enddate_fact,
        (after).amt_credit_max_overdue AS amt_credit_max_overdue,
        (after).cnt_credit_prolong AS cnt_credit_prolong,
        (after).amt_credit_sum AS amt_credit_sum,
        (after).amt_credit_sum_debt AS amt_credit_sum_debt,
        (after).amt_credit_sum_limit AS amt_credit_sum_limit,
        (after).amt_credit_sum_overdue AS amt_credit_sum_overdue,
        (after).credit_type AS credit_type,
        (after).days_credit_update AS days_credit_update,
        (after).amt_annuity AS amt_annuity,
        ts_ms,
        event_time
      FROM bureau_raw
      WHERE op IN ('c','u')
    """

flink_flatten_bureau_balance_table = """
      SELECT
        (after).sk_id_bureau AS sk_id_bureau,
        (after).months_balance AS months_balance,
        (after).status AS status,
        ts_ms,
        event_time
      FROM bureau_balance_raw
      WHERE op IN ('c','u')
    """

output_table = f"""
    CREATE TABLE merged_out (
      sk_id_bureau BIGINT,
      sk_id_curr BIGINT,
      months_balance INT,
      status STRING,
      updated STRING,
      PRIMARY KEY (sk_id_curr) NOT ENFORCED
    ) WITH (
      'connector' = 'kafka',
      'topic' = '{OUTPUT_TOPIC}',
      'properties.bootstrap.servers' = '{BOOTSTRAP}',
      'key.format' = 'json',                  
      'value.format' = 'json',               
      'json.ignore-parse-errors' = 'true',
    )
    """
