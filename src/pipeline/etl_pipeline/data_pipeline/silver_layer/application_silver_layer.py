from pyspark.sql import SparkSession
from pyspark.sql import functions as F

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    MINIO_ENDPOINT,
    MINIO_ACCESS_KEY,
    MINIO_SECRET_KEY,
    SILVER_PATH_APPLICATION,
    write_with_delta_sql,
    add_scd_type2_cols
)

from silver_etl_schema import (
    silver_fact_schema,
    silver_dim_user_demographic_schema,
    silver_dim_user_contact_schema,
    silver_dim_user_region_schema,
    silver_dim_asset_schema,
    silver_dim_user_income_schema,
    silver_dim_external_source_schema,
    silver_dim_application_time_schema,
    silver_dim_provided_docs_schema,
    silver_dim_aggregated_schema,
)

def main():
    spark = (
        SparkSession.builder
        .appName("Silver application batching")
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

    
    bronze_table = spark.read.table("homecredit_bronze.application")
    
    silver_fact = bronze_table.select(
        "sk_id_curr",
        "amt_income_total",
        "amt_credit",
        "amt_annuity",
        "amt_goods_price",
        "amt_req_credit_bureau_hour",
        "amt_req_credit_bureau_day",
        "amt_req_credit_bureau_week",
        "amt_req_credit_bureau_mon",
        "amt_req_credit_bureau_qrt",
        "amt_req_credit_bureau_year",
        "event_ts"
    )

    silver_dim_user_demographic = (
        bronze_table.select(
            "sk_id_curr",
            "cnt_children",
            "cnt_fam_members",
            "occupation_type",
            "organization_type",
            "days_birth",
            "days_employed",
            "event_ts"
        )
        .withColumn("age_years", F.floor(F.abs(F.col("days_birth")) / F.lit(365)))
        .withColumn(
            "years_employed",
            F.when(
                F.col("days_employed").isNotNull(),
                F.abs(F.col("days_employed")) / F.lit(365.25),
            ),
        )
    )
    
    silver_dim_user_contact = bronze_table.select(
        "sk_id_curr",
        "flag_mobil",
        "flag_emp_phone",
        "flag_work_phone",
        "flag_cont_mobile",
        "flag_phone",
        "flag_email",
        "days_last_phone_change",
        "event_ts",
    ).withColumn(
        "days_last_phone_change",
        F.to_date(
            F.date_add(F.current_date(), F.col("days_last_phone_change").cast("int"))
        ),
    )
    
    silver_dim_user_region = bronze_table.select(
        "sk_id_curr", 
        "region_population_relative",
        "region_rating_client",
        "region_rating_client_w_city",
        "reg_region_not_live_region",
        "reg_region_not_work_region",
        "live_region_not_work_region",
        "reg_city_not_live_city",
        "reg_city_not_work_city",
        "live_city_not_work_city",
        "event_ts"
    )
    
    silver_dim_asset = bronze_table.select(
        "sk_id_curr",
        "flag_own_car",
        "flag_own_realty",
        "own_car_age",
        "name_housing_type",
        "name_type_suite",
        "event_ts"
    )
    
    silver_dim_user_income = bronze_table.select(
        "sk_id_curr",
        "amt_income_total",
        "name_contract_type",
        "name_income_type",
        "name_education_type",
        "name_family_status",
        "amt_credit",
        "amt_annuity",
        "amt_goods_price",
        "event_ts"
    )
    
    silver_dim_external_source = bronze_table.select(
        "sk_id_curr",
        "ext_source_1",
        "ext_source_2",
        "ext_source_3",
        "event_ts"
    )
    
    silver_dim_application_time = bronze_table.select(
        "sk_id_curr",
        "days_registration",
        "event_ts",
        "days_id_publish",
        "weekday_appr_process_start",
        "hour_appr_process_start"
    ).withColumn(
        "days_registration",
        F.to_date(F.date_add(F.current_date(), F.col("days_registration").cast("int")))
    ).withColumn(
        "days_id_publish",
        F.to_date(F.date_add(F.current_date(), F.col("days_id_publish").cast("int")))
    ).withColumn(
        "is_weekend",
        F.when(F.col("weekday_appr_process_start").isin("SATURDAY", "SUNDAY"), F.lit(1)).otherwise(F.lit(0))
    ).withColumn(
        "is_working_hour",
        F.when(F.col("hour_appr_process_start").between(9, 18), F.lit(1)).otherwise(F.lit(0))
    )

    docs_cols = [f"flag_document_{i}" for i in range(2,22)]

    aggregated_cols = [
        col for col in bronze_table.columns if col.endswith(("_avg", "_max", "_min", "_medi", "_mode"))
    ]

    silver_dim_provided_docs = bronze_table.select(
        "sk_id_curr",
        *docs_cols,
        "event_ts",
    )


    silver_dim_aggregated = bronze_table.select(
        "sk_id_curr",
        *aggregated_cols,
        "obs_30_cnt_social_circle",
        "obs_60_cnt_social_circle",
        "def_30_cnt_social_circle",
        "def_60_cnt_social_circle",
        "event_ts",
    )
    
    dim_tables = {
        "dim_user_demographic": silver_dim_user_demographic,
        "dim_user_contact": silver_dim_user_contact,
        "dim_user_region": silver_dim_user_region,
        "dim_asset": silver_dim_asset,
        "dim_user_income": silver_dim_user_income,
        "dim_external_source": silver_dim_external_source,
        "dim_application_time": silver_dim_application_time,
        "dim_provided_docs": silver_dim_provided_docs,
        "dim_aggregated": silver_dim_aggregated,
    }
    
    dim_tables_schemas = {
        "dim_user_demographic": silver_dim_user_demographic_schema,
        "dim_user_contact": silver_dim_user_contact_schema,
        "dim_user_region": silver_dim_user_region_schema,
        "dim_asset": silver_dim_asset_schema,
        "dim_user_income": silver_dim_user_income_schema,
        "dim_external_source": silver_dim_external_source_schema,
        "dim_application_time": silver_dim_application_time_schema,
        "dim_provided_docs": silver_dim_provided_docs_schema,
        "dim_aggregated": silver_dim_aggregated_schema,
    }
    
    spark.sql("CREATE SCHEMA IF NOT EXISTS homecredit_silver")

    #  Fact table 
    write_with_delta_sql(
        spark,
        silver_fact,
        table_name="fact_loan",
        schema="homecredit_silver",
        base_path=SILVER_PATH_APPLICATION,
        table_ddl=silver_fact_schema,
        mode="append"
    )

    #  Dim tables 
    for name, df in dim_tables.items():
        schema = dim_tables_schemas[name]

        df_scd2 = add_scd_type2_cols(df, ts_col="event_ts")

        write_with_delta_sql(
            spark,
            df_scd2,
            table_name=name,
            schema="homecredit_silver",
            base_path=SILVER_PATH_APPLICATION,
            table_ddl=schema,
            mode="append"
        )


            
    
if __name__ == "__main__":
    main()
