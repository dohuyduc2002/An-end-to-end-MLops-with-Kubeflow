from pyspark.sql.types import (
    LongType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    DateType, 
    TimestampType
)

gold_fact_loan_schema = StructType(
    [
        StructField("sk_id_curr", LongType()),
        StructField("amt_income_total", DoubleType()),
        StructField("amt_credit", DoubleType()),
        StructField("amt_annuity", DoubleType()),
        StructField("amt_goods_price", DoubleType()),
        StructField("amt_req_credit_bureau_hour", DoubleType()),
        StructField("amt_req_credit_bureau_day", DoubleType()),
        StructField("amt_req_credit_bureau_week", DoubleType()),
        StructField("amt_req_credit_bureau_mon", DoubleType()),
        StructField("amt_req_credit_bureau_qrt", DoubleType()),
        StructField("amt_req_credit_bureau_year", DoubleType()),
        StructField("event_ts", TimestampType())
    ]
)

gold_fact_bureau_balance = StructType(
    [
        StructField("sk_id_bureau", LongType(), True),
        StructField("sk_id_curr", LongType(), True),
        StructField("months_balance", IntegerType(), True),
        StructField("status", StringType(), True),
        StructField("updated", StringType(), True),
        StructField("event_ts", TimestampType())
    ]
)

gold_dim_user_demographic_schema = StructType(
    [
        StructField("sk_id_curr", LongType()),
        StructField("cnt_children", IntegerType()),
        StructField("cnt_fam_members", DoubleType()),
        StructField("occupation_type", StringType()),
        StructField("organization_type", StringType()),
        StructField("days_birth", IntegerType()),
        StructField("days_employed", IntegerType()),
        StructField("age_years", LongType()),
        StructField("years_employed", DoubleType()),
        StructField("effective_date", TimestampType())
    ]
)

gold_dim_user_contact_schema = StructType(
    [
        StructField("sk_id_curr", LongType()),
        StructField("flag_mobil", IntegerType()),
        StructField("flag_emp_phone", IntegerType()),
        StructField("flag_work_phone", IntegerType()),
        StructField("flag_cont_mobile", IntegerType()),
        StructField("flag_phone", IntegerType()),
        StructField("flag_email", IntegerType()),
        StructField("days_last_phone_change", DateType()),
        StructField("effective_date", TimestampType())
    ]
)

gold_dim_user_region_schema = StructType(
    [
        StructField("sk_id_curr", LongType()),
        StructField("region_population_relative", DoubleType()),
        StructField("region_rating_client", IntegerType()),
        StructField("region_rating_client_w_city", IntegerType()),
        StructField("reg_region_not_live_region", IntegerType()),
        StructField("reg_region_not_work_region", IntegerType()),
        StructField("live_region_not_work_region", IntegerType()),
        StructField("reg_city_not_live_city", IntegerType()),
        StructField("reg_city_not_work_city", IntegerType()),
        StructField("live_city_not_work_city", IntegerType()),
        StructField("effective_date", TimestampType())
    ]
)

gold_dim_asset_and_bureau_schema = StructType(
    [
        StructField("sk_id_curr", LongType()),
        StructField("flag_own_car", StringType()),
        StructField("flag_own_realty", StringType()),
        StructField("name_housing_type", StringType()),
        StructField("name_type_suite", StringType()),
        StructField("own_car_age", DoubleType()),
        StructField("sk_id_bureau", StringType()),
        StructField("credit_active", StringType()),
        StructField("credit_currency", StringType()),
        StructField("days_credit", StringType()),
        StructField("credit_day_overdue", StringType()),
        StructField("days_credit_enddate", StringType()),
        StructField("days_enddate_fact", StringType()),
        StructField("amt_credit_max_overdue", StringType()),
        StructField("cnt_credit_prolong", StringType()),
        StructField("amt_credit_sum", StringType()),
        StructField("amt_credit_sum_debt", StringType()),
        StructField("amt_credit_sum_limit", StringType()),
        StructField("amt_credit_sum_overdue", StringType()),
        StructField("credit_type", StringType()),
        StructField("days_credit_update", StringType()),
        StructField("amt_annuity", StringType()),
        StructField("effective_date", TimestampType())
    ]
)
gold_dim_user_income_schema = StructType(
    [
        StructField("sk_id_curr", LongType()),
        StructField("amt_income_total", DoubleType()),
        StructField("name_contract_type", StringType()),
        StructField("name_income_type", StringType()),
        StructField("name_education_type", StringType()),
        StructField("name_family_status", StringType()),
        StructField("amt_credit", DoubleType()),
        StructField("amt_annuity", DoubleType()),
        StructField("amt_goods_price", DoubleType()),
        StructField("effective_date", TimestampType())
    ]
)

gold_dim_external_source_schema = StructType(
    [
        StructField("sk_id_curr", LongType()),
        StructField("ext_source_1", DoubleType()),
        StructField("ext_source_2", DoubleType()),
        StructField("ext_source_3", DoubleType()),
        StructField("effective_date", TimestampType())
    ]
)

gold_dim_application_time_schema = StructType(
    [
        StructField("sk_id_curr", LongType()),
        StructField("days_registration", DoubleType()),
        StructField("days_id_publish", IntegerType()),
        StructField("hour_appr_process_start", IntegerType()),
        StructField("weekday_appr_process_start", StringType()),
        StructField("days_id_publish", StringType()),
        StructField("is_weekend", IntegerType()),
        StructField("is_weekend", IntegerType()),
        StructField("is_working_hour", IntegerType()),
        StructField("effective_date", TimestampType())
    ]
)

gold_dim_provided_docs_schema = StructType(
    [
        StructField("sk_id_curr", LongType()),
        StructField("flag_document_2", IntegerType()),
        StructField("flag_document_3", IntegerType()),
        StructField("flag_document_4", IntegerType()),
        StructField("flag_document_5", IntegerType()),
        StructField("flag_document_6", IntegerType()),
        StructField("flag_document_7", IntegerType()),
        StructField("flag_document_8", IntegerType()),
        StructField("flag_document_9", IntegerType()),
        StructField("flag_document_10", IntegerType()),
        StructField("flag_document_11", IntegerType()),
        StructField("flag_document_12", IntegerType()),
        StructField("flag_document_13", IntegerType()),
        StructField("flag_document_14", IntegerType()),
        StructField("flag_document_15", IntegerType()),
        StructField("flag_document_16", IntegerType()),
        StructField("flag_document_17", IntegerType()),
        StructField("flag_document_18", IntegerType()),
        StructField("flag_document_19", IntegerType()),
        StructField("flag_document_20", IntegerType()),
        StructField("flag_document_21", IntegerType()),
        StructField("effective_date", TimestampType())
    ]
)

gold_dim_aggregated_schema = StructType([
    StructField("sk_id_curr", LongType()),
    StructField("apartments_avg", DoubleType()),
    StructField("basementarea_avg", DoubleType()),
    StructField("years_beginexpluatation_avg", DoubleType()),
    StructField("years_build_avg", DoubleType()),
    StructField("commonarea_avg", DoubleType()),
    StructField("elevators_avg", DoubleType()),
    StructField("entrances_avg", DoubleType()),
    StructField("floorsmax_avg", DoubleType()),
    StructField("floorsmin_avg", DoubleType()),
    StructField("landarea_avg", DoubleType()),
    StructField("livingapartments_avg", DoubleType()),
    StructField("livingarea_avg", DoubleType()),
    StructField("nonlivingapartments_avg", DoubleType()),
    StructField("nonlivingarea_avg", DoubleType()),
    StructField("apartments_mode", DoubleType()),
    StructField("basementarea_mode", DoubleType()),
    StructField("years_beginexpluatation_mode", DoubleType()),
    StructField("years_build_mode", DoubleType()),
    StructField("commonarea_mode", DoubleType()),
    StructField("elevators_mode", DoubleType()),
    StructField("entrances_mode", DoubleType()),
    StructField("floorsmax_mode", DoubleType()),
    StructField("floorsmin_mode", DoubleType()),
    StructField("landarea_mode", DoubleType()),
    StructField("livingapartments_mode", DoubleType()),
    StructField("livingarea_mode", DoubleType()),
    StructField("nonlivingapartments_mode", DoubleType()),
    StructField("nonlivingarea_mode", DoubleType()),
    StructField("apartments_medi", DoubleType()),
    StructField("basementarea_medi", DoubleType()),
    StructField("years_beginexpluatation_medi", DoubleType()),
    StructField("years_build_medi", DoubleType()),
    StructField("commonarea_medi", DoubleType()),
    StructField("elevators_medi", DoubleType()),
    StructField("entrances_medi", DoubleType()),
    StructField("floorsmax_medi", DoubleType()),
    StructField("floorsmin_medi", DoubleType()),
    StructField("landarea_medi", DoubleType()),
    StructField("livingapartments_medi", DoubleType()),
    StructField("livingarea_medi", DoubleType()),
    StructField("nonlivingapartments_medi", DoubleType()),
    StructField("nonlivingarea_medi", DoubleType()),
    StructField("fondkapremont_mode", StringType()),
    StructField("housetype_mode", StringType()),
    StructField("totalarea_mode", DoubleType()),
    StructField("wallsmaterial_mode", StringType()),
    StructField("emergencystate_mode", StringType()),
    StructField("obs_30_cnt_social_circle", DoubleType()),
    StructField("def_30_cnt_social_circle", DoubleType()),
    StructField("obs_60_cnt_social_circle", DoubleType()),
    StructField("def_60_cnt_social_circle", DoubleType()),
    StructField("effective_date", TimestampType())
])

gold_fact_bureau_balance_schema = StructType(
    [
        StructField("sk_id_bureau", LongType(), True),
        StructField("sk_id_curr", LongType(), True),
        StructField("months_balance", IntegerType(), True),
        StructField("status", StringType(), True),
        StructField("updated", StringType(), True),
        StructField("effective_date", TimestampType())
    ]
)