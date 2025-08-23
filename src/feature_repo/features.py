from datetime import timedelta

import pandas as pd
from feast import (
    FeatureView,
    Field,
)
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float64, String, Int32, Int64

from data_sources import create_spark_source, create_kafka_source
from entities import application_entity, bureau_balance_entity

spark_src_name = "application"
spark_path = "s3a://sample-data/curated_application"

kafka_src_name = "merged-bureau-with-sk-id-curr"
kafka_src_topic = "flink-merged-bureau"
kafka_bootstrap_servers = "kafka-cluster-0-kafka-bootstrap.kafka.svc.cluster.local:9092"
stream_schema = """
sk_id_bureau BIGINT,
sk_id_curr BIGINT,
months_balance INT,
status STRING,
updated TIMESTAMP
"""

spark_source = create_spark_source(spark_src_name, spark_path)
stream_source = create_kafka_source(kafka_src_name, kafka_src_topic, kafka_bootstrap_servers, stream_schema)

application_view = FeatureView(
    name="application",
    description="Application for loan",
    entities=[application_entity],
    ttl=timedelta(seconds=8640000000),
    schema=[
        Field(name="sk_id_curr", dtype=Int64),
        Field(name="target", dtype=Int32),
        Field(name="name_contract_type", dtype=String),
        Field(name="code_gender", dtype=String),
        Field(name="flag_own_car", dtype=String),
        Field(name="flag_own_realty", dtype=String),
        Field(name="cnt_children", dtype=Int32),
        Field(name="amt_income_total", dtype=Float64),
        Field(name="amt_credit", dtype=Float64),
        Field(name="amt_annuity", dtype=Float64),
        Field(name="amt_goods_price", dtype=Float64),
        Field(name="name_type_suite", dtype=String),
        Field(name="name_income_type", dtype=String),
        Field(name="name_education_type", dtype=String),
        Field(name="name_family_status", dtype=String),
        Field(name="name_housing_type", dtype=String),
        Field(name="region_population_relative", dtype=Float64),
        Field(name="days_birth", dtype=Int32),
        Field(name="days_employed", dtype=Int32),
        Field(name="days_registration", dtype=Float64),
        Field(name="days_id_publish", dtype=Int32),
        Field(name="own_car_age", dtype=Float64),
        Field(name="flag_mobil", dtype=Int32),
        Field(name="flag_emp_phone", dtype=Int32),
        Field(name="flag_work_phone", dtype=Int32),
        Field(name="flag_cont_mobile", dtype=Int32),
        Field(name="flag_phone", dtype=Int32),
        Field(name="flag_email", dtype=Int32),
        Field(name="occupation_type", dtype=String),
        Field(name="cnt_fam_members", dtype=Float64),
        Field(name="region_rating_client", dtype=Int32),
        Field(name="region_rating_client_w_city", dtype=Int32),
        Field(name="weekday_appr_process_start", dtype=String),
        Field(name="hour_appr_process_start", dtype=Int32),
        Field(name="reg_region_not_live_region", dtype=Int32),
        Field(name="reg_region_not_work_region", dtype=Int32),
        Field(name="live_region_not_work_region", dtype=Int32),
        Field(name="reg_city_not_live_city", dtype=Int32),
        Field(name="reg_city_not_work_city", dtype=Int32),
        Field(name="live_city_not_work_city", dtype=Int32),
        Field(name="organization_type", dtype=String),
        Field(name="ext_source_1", dtype=Float64),
        Field(name="ext_source_2", dtype=Float64),
        Field(name="ext_source_3", dtype=Float64),
        Field(name="apartments_avg", dtype=Float64),
        Field(name="basementarea_avg", dtype=Float64),
        Field(name="years_beginexpluatation_avg", dtype=Float64),
        Field(name="years_build_avg", dtype=Float64),
        Field(name="commonarea_avg", dtype=Float64),
        Field(name="elevators_avg", dtype=Float64),
        Field(name="entrances_avg", dtype=Float64),
        Field(name="floorsmax_avg", dtype=Float64),
        Field(name="floorsmin_avg", dtype=Float64),
        Field(name="landarea_avg", dtype=Float64),
        Field(name="livingapartments_avg", dtype=Float64),
        Field(name="livingarea_avg", dtype=Float64),
        Field(name="nonlivingapartments_avg", dtype=Float64),
        Field(name="nonlivingarea_avg", dtype=Float64),
        Field(name="apartments_mode", dtype=Float64),
        Field(name="basementarea_mode", dtype=Float64),
        Field(name="years_beginexpluatation_mode", dtype=Float64),
        Field(name="years_build_mode", dtype=Float64),
        Field(name="commonarea_mode", dtype=Float64),
        Field(name="elevators_mode", dtype=Float64),
        Field(name="entrances_mode", dtype=Float64),
        Field(name="floorsmax_mode", dtype=Float64),
        Field(name="floorsmin_mode", dtype=Float64),
        Field(name="landarea_mode", dtype=Float64),
        Field(name="livingapartments_mode", dtype=Float64),
        Field(name="livingarea_mode", dtype=Float64),
        Field(name="nonlivingapartments_mode", dtype=Float64),
        Field(name="nonlivingarea_mode", dtype=Float64),
        Field(name="apartments_medi", dtype=Float64),
        Field(name="basementarea_medi", dtype=Float64),
        Field(name="years_beginexpluatation_medi", dtype=Float64),
        Field(name="years_build_medi", dtype=Float64),
        Field(name="commonarea_medi", dtype=Float64),
        Field(name="elevators_medi", dtype=Float64),
        Field(name="entrances_medi", dtype=Float64),
        Field(name="floorsmax_medi", dtype=Float64),
        Field(name="floorsmin_medi", dtype=Float64),
        Field(name="landarea_medi", dtype=Float64),
        Field(name="livingapartments_medi", dtype=Float64),
        Field(name="livingarea_medi", dtype=Float64),
        Field(name="nonlivingapartments_medi", dtype=Float64),
        Field(name="nonlivingarea_medi", dtype=Float64),
        Field(name="fondkapremont_mode", dtype=String),
        Field(name="housetype_mode", dtype=String),
        Field(name="totalarea_mode", dtype=Float64),
        Field(name="wallsmaterial_mode", dtype=String),
        Field(name="emergencystate_mode", dtype=String),
        Field(name="obs_30_cnt_social_circle", dtype=Float64),
        Field(name="def_30_cnt_social_circle", dtype=Float64),
        Field(name="obs_60_cnt_social_circle", dtype=Float64),
        Field(name="def_60_cnt_social_circle", dtype=Float64),
        Field(name="days_last_phone_change", dtype=Float64),
        Field(name="flag_document_2", dtype=Int32),
        Field(name="flag_document_3", dtype=Int32),
        Field(name="flag_document_4", dtype=Int32),
        Field(name="flag_document_5", dtype=Int32),
        Field(name="flag_document_6", dtype=Int32),
        Field(name="flag_document_7", dtype=Int32),
        Field(name="flag_document_8", dtype=Int32),
        Field(name="flag_document_9", dtype=Int32),
        Field(name="flag_document_10", dtype=Int32),
        Field(name="flag_document_11", dtype=Int32),
        Field(name="flag_document_12", dtype=Int32),
        Field(name="flag_document_13", dtype=Int32),
        Field(name="flag_document_14", dtype=Int32),
        Field(name="flag_document_15", dtype=Int32),
        Field(name="flag_document_16", dtype=Int32),
        Field(name="flag_document_17", dtype=Int32),
        Field(name="flag_document_18", dtype=Int32),
        Field(name="flag_document_19", dtype=Int32),
        Field(name="flag_document_20", dtype=Int32),
        Field(name="flag_document_21", dtype=Int32),
        Field(name="amt_req_credit_bureau_hour", dtype=Float64),
        Field(name="amt_req_credit_bureau_day", dtype=Float64),
        Field(name="amt_req_credit_bureau_week", dtype=Float64),
        Field(name="amt_req_credit_bureau_mon", dtype=Float64),
        Field(name="amt_req_credit_bureau_qrt", dtype=Float64),
        Field(name="amt_req_credit_bureau_year", dtype=Float64),
        Field(name="updated", dtype=Float64),
    ],
    online=True,
    source=spark_source,
    tags={"development": "True"},
    owner="dohuyduc.work@gmail.com",
)


merged_bureau_balance_view = FeatureView(
    name="merged-bureau-with-sk-id-curr",
    description="Merged bureau balance with sk_id_curr",
    entities=[bureau_balance_entity],
    ttl=timedelta(seconds=8640000000),
    schema=[
        Field(name="sk_id_bureau", dtype=String),
        Field(name="sk_id_curr", dtype=String),
        Field(name="months_balance", dtype=Int32),
        Field(name="status", dtype=String),
        Field(name="updated", dtype=String),
    ],
    online=True,
    source=stream_source,
    tags={"development": "True"},
    owner="dohuyduc.work@gmail.com",
)

# Define an on demand feature view which can generate new features based on
# existing feature views and RequestSource features
# this will use in KFP
# @on_demand_feature_view(
#     sources=[merged_bureau_balance_view],
#     schema=[
#         Field(name="sk_id_bureau", dtype=String),
#         Field(name="sk_id_curr", dtype=String),
#         Field(name="months_balance", dtype=Int32),
#         Field(name="status", dtype=String),
#         Field(name="updated", dtype=String),
#     ],
#     entities=[bureau_balance_entity],
#     owner="dohuyduc.work@gmail.com",
#     tags={"development": "True"}
# )
