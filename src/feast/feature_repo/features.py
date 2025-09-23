from datetime import timedelta

import pandas as pd
from feast import (
    FeatureView,
    Field,
)
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float64, String, Int32, Int64, Double, UnixTimestamp

from data_sources import create_spark_source, create_kafka_source
from entities import *

spark_src = {
    "gold_fact_loan": "s3a://data-mart/gold_fact_loan",
    "gold_fact_bureau_balance": "s3a://data-mart/gold_fact_bureau_balance",
    "gold_dim_demographic": "s3a://data-mart/gold_dim_demographic",
    "gold_dim_user_contract": "s3a://data-mart/gold_dim_user_contract",
    "gold_dim_user_region": "s3a://data-mart/gold_dim_user_region",
    "gold_dim_asset_and_bureau": "s3a://data-mart/gold_dim_asset_and_bureau",
    "gold_dim_user_income": "s3a://data-mart/gold_dim_user_income",
    "gold_dim_external_source": "s3a://data-mart/gold_dim_external_source",
    "gold_dim_application_time": "s3a://data-mart/gold_dim_application_time",
    "gold_dim_provided_docs": "s3a://data-mart/gold_dim_provided_docs",
    "gold_dim_aggregated": "s3a://data-mart/gold_dim_aggregated",
}

spark_sources = {}

for name, path in spark_src.items():
    spark_sources[name] = create_spark_source(name, path)

###### 

kafka_src_name = "raw-bureau-balance-change"
kafka_src_topic = "flink-merged-bureau"
kafka_bootstrap_servers = "kafka-cluster-0-kafka-bootstrap.kafka.svc.cluster.local:9092"

stream_schema = "sk_id_bureau int64, sk_id_curr int64, months_balance int32, status string, updated string"
stream_source = create_kafka_source(kafka_src_name, kafka_src_topic, kafka_bootstrap_servers, stream_schema, None)

loan_view = FeatureView(
    name="application",
    description="Application features",
    entities=[gold_fact_loan_entity],
    ttl=timedelta(days=60),
    schema=[
        Field(name="sk_id_curr", dtype=Int64),
        Field(name="amt_income_total", dtype=Float64),
        Field(name="amt_credit", dtype=Float64),
        Field(name="amt_annuity", dtype=Float64),
        Field(name="amt_goods_price", dtype=Float64),
        Field(name="amt_req_credit_bureau_hour", dtype=Float64),
        Field(name="amt_req_credit_bureau_day", dtype=Float64),
        Field(name="amt_req_credit_bureau_week", dtype=Float64),
        Field(name="amt_req_credit_bureau_mon", dtype=Float64),
        Field(name="amt_req_credit_bureau_qrt", dtype=Float64),
        Field(name="amt_req_credit_bureau_year", dtype=Float64),
    ],
    source=spark_sources["gold_fact_loan"],
    online=True,
    timestamp_field="event_ts",
    tags={"development": "True"},
    owner="dohuyduc.work@gmail.com",
)

bureau_balance_view = FeatureView(
    name="bureau_balance",
    description="Bureau balance history features",
    entities=[gold_fact_bureau_balance_entity],  
    ttl=timedelta(days=60),
    schema=[
        Field(name="sk_id_bureau", dtype=Int64),
        Field(name="sk_id_curr", dtype=Int64),
        Field(name="months_balance", dtype=Int32),
        Field(name="status", dtype=String),
        Field(name="updated", dtype=String),
    ],
    source=spark_sources["gold_fact_bureau_balance"],
    timestamp_field="effective_date",
    tags={"development": "True"},
    owner="dohuyduc.work@gmail.com",
)

user_contact_view = FeatureView(
    name="user_contact",
    description="User contact and communication features",
    entities=[gold_dim_user_contract_entity], 
    ttl=timedelta(days=60),
    schema=[
        Field(name="sk_id_curr", dtype=Int64),
        Field(name="flag_mobil", dtype=Int32),
        Field(name="flag_emp_phone", dtype=Int32),
        Field(name="flag_work_phone", dtype=Int32),
        Field(name="flag_cont_mobile", dtype=Int32),
        Field(name="flag_phone", dtype=Int32),
        Field(name="flag_email", dtype=Int32),
        Field(name="days_last_phone_change", dtype=UnixTimestamp),  
    ],
    source=spark_sources["gold_dim_user_contact"],
    timestamp_field="effective_date",
    tags={"development": "True"},
    owner="dohuyduc.work@gmail.com",
)

user_demographic_view = FeatureView(
    name="user_demographic",
    description="User demographic features",
    entities=[gold_dim_demographic_entity],  
    ttl=timedelta(days=60),
    schema=[
        Field(name="sk_id_curr", dtype=Int64),
        Field(name="cnt_children", dtype=Int32),
        Field(name="cnt_fam_members", dtype=Float64),
        Field(name="occupation_type", dtype=String),
        Field(name="organization_type", dtype=String),
        Field(name="days_birth", dtype=Int32),
        Field(name="days_employed", dtype=Int32),
        Field(name="age_years", dtype=Int64),
        Field(name="years_employed", dtype=Float64),
    ],
    source=spark_sources["gold_dim_demographic"],
    timestamp_field="effective_date",
    tags={"development": "True"},
    owner="dohuyduc.work@gmail.com",
)

user_contact_view = FeatureView(
    name="user_contact",
    description="User contact and communication features",
    entities=[gold_dim_user_contract_entity],  
    ttl=timedelta(days=60),
    schema=[
        Field(name="sk_id_curr", dtype=Int64),
        Field(name="flag_mobil", dtype=Int32),
        Field(name="flag_emp_phone", dtype=Int32),
        Field(name="flag_work_phone", dtype=Int32),
        Field(name="flag_cont_mobile", dtype=Int32),
        Field(name="flag_phone", dtype=Int32),
        Field(name="flag_email", dtype=Int32),
        Field(name="days_last_phone_change", dtype=String),
        Field(name="days_last_phone_change", dtype=UnixTimestamp),
    ],
    source=spark_sources["gold_dim_user_contact"],
    timestamp_field="effective_date",
    tags={"development": "True"},
    owner="dohuyduc.work@gmail.com",
)

user_region_view = FeatureView(
    name="user_region",
    description="User regional and city mismatch features",
    entities=[gold_dim_user_region_entity],  
    ttl=timedelta(days=60),
    schema=[
        Field(name="sk_id_curr", dtype=Int64),
        Field(name="region_population_relative", dtype=Float64),
        Field(name="region_rating_client", dtype=Int32),
        Field(name="region_rating_client_w_city", dtype=Int32),
        Field(name="reg_region_not_live_region", dtype=Int32),
        Field(name="reg_region_not_work_region", dtype=Int32),
        Field(name="live_region_not_work_region", dtype=Int32),
        Field(name="reg_city_not_live_city", dtype=Int32),
        Field(name="reg_city_not_work_city", dtype=Int32),
        Field(name="live_city_not_work_city", dtype=Int32),
    ],
    source=spark_sources["gold_dim_user_region"],
    timestamp_field="effective_date",
    tags={"development": "True"},
    owner="dohuyduc.work@gmail.com",
)

asset_and_bureau_view = FeatureView(
    name="asset_and_bureau",
    description="User assets and bureau credit features",
    entities=[gold_dim_asset_and_bureau_entity],  
    ttl=timedelta(days=60),
    schema=[
        Field(name="sk_id_curr", dtype=Int64),
        Field(name="flag_own_car", dtype=String),
        Field(name="flag_own_realty", dtype=String),
        Field(name="name_housing_type", dtype=String),
        Field(name="name_type_suite", dtype=String),
        Field(name="own_car_age", dtype=Float64),
        Field(name="sk_id_bureau", dtype=String),
        Field(name="credit_active", dtype=String),
        Field(name="credit_currency", dtype=String),
        Field(name="days_credit", dtype=String),
        Field(name="credit_day_overdue", dtype=String),
        Field(name="days_credit_enddate", dtype=String),
        Field(name="days_enddate_fact", dtype=String),
        Field(name="amt_credit_max_overdue", dtype=String),
        Field(name="cnt_credit_prolong", dtype=String),
        Field(name="amt_credit_sum", dtype=String),
        Field(name="amt_credit_sum_debt", dtype=String),
        Field(name="amt_credit_sum_limit", dtype=String),
        Field(name="amt_credit_sum_overdue", dtype=String),
        Field(name="credit_type", dtype=String),
        Field(name="days_credit_update", dtype=String),
        Field(name="amt_annuity", dtype=String),
    ],
    source=spark_sources["gold_dim_asset_and_bureau"],
    timestamp_field="effective_date",
    tags={"development": "True"},
    owner="dohuyduc.work@gmail.com",
)

user_income_view = FeatureView(
    name="user_income",
    description="User income, contract, and credit-related features",
    entities=[gold_dim_user_income_entity],  
    ttl=timedelta(days=60),
    schema=[
        Field(name="sk_id_curr", dtype=Int64),
        Field(name="amt_income_total", dtype=Float64),
        Field(name="name_contract_type", dtype=String),
        Field(name="name_income_type", dtype=String),
        Field(name="name_education_type", dtype=String),
        Field(name="name_family_status", dtype=String),
        Field(name="amt_credit", dtype=Float64),
        Field(name="amt_annuity", dtype=Float64),
        Field(name="amt_goods_price", dtype=Float64),
    ],
    source=spark_sources["gold_dim_user_income"],
    timestamp_field="effective_date",
    tags={"development": "True"},
    owner="dohuyduc.work@gmail.com",
)

external_source_view = FeatureView(
    name="external_source",
    description="External score source features",
    entities=[gold_dim_external_source_entity], 
    ttl=timedelta(days=60),
    schema=[
        Field(name="sk_id_curr", dtype=Int64),
        Field(name="ext_source_1", dtype=Float64),
        Field(name="ext_source_2", dtype=Float64),
        Field(name="ext_source_3", dtype=Float64),
    ],
    source=spark_sources["gold_dim_external_source"],
    timestamp_field="effective_date",
    tags={"development": "True"},
    owner="dohuyduc.work@gmail.com",
)

application_time_view = FeatureView(
    name="application_time",
    description="Application timing features",
    entities=[gold_dim_application_time_entity], 
    ttl=timedelta(days=60),
    schema=[
        Field(name="sk_id_curr", dtype=Int64),
        Field(name="days_registration", dtype=UnixTimestamp),  
        Field(name="days_id_publish", dtype=UnixTimestamp),    
        Field(name="hour_appr_process_start", dtype=Int32),
        Field(name="weekday_appr_process_start", dtype=String),
        Field(name="is_weekend", dtype=Int32),
        Field(name="is_working_hour", dtype=Int32),
    ],
    source=spark_sources["gold_dim_application_time"],
    timestamp_field="effective_date",
    tags={"development": "True"},
    owner="dohuyduc.work@gmail.com",
)

provided_docs_view = FeatureView(
    name="provided_docs",
    description="Flags for user provided documents",
    entities=[gold_dim_provided_docs_entity],  
    ttl=timedelta(days=60),
    schema=[
        Field(name="sk_id_curr", dtype=Int64),
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
    ],
    source=spark_sources["gold_dim_provided_docs"],
    timestamp_field="effective_date",
    tags={"development": "True"},
    owner="dohuyduc.work@gmail.com",
)

aggregated_view = FeatureView(
    name="aggregated",
    description="Aggregated features from multiple domains",
    entities=[gold_dim_aggregated_entity], 
    ttl=timedelta(days=60),
    schema=[
        Field(name="sk_id_curr", dtype=Int64),
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
    ],
    source=spark_sources["gold_dim_aggregated"],
    timestamp_field="effective_date",
    tags={"development": "True"},
    owner="dohuyduc.work@gmail.com",
)
#####

stream_flink_merged_bureau_balance_view = FeatureView(
    name="stream_flink_merged_bureau_balance",
    description="Merged bureau balance with sk_id_curr",
    entities=[stream_bureau_balance_entity],
    ttl=timedelta(days=60),
    schema=[
        Field(name="sk_id_bureau", dtype=String),
        Field(name="sk_id_curr", dtype=String),
        Field(name="months_balance", dtype=Int32),
        Field(name="status", dtype=String),
        Field(name="updated", dtype=String),
    ],
    source=stream_source,
    online=True,
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
