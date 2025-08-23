from pyspark.sql import SparkSession
from pyspark.sql import functions as F

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    MINIO_ENDPOINT,
    MINIO_ACCESS_KEY,
    MINIO_SECRET_KEY,
    GOLD_PATH,
    SILVER_PATH_APPLICATION,
)

def main():
    def write_delta_overwrite(df, table_name):
        path = f"s3a://{GOLD_PATH}/{table_name}"
        (
            df.write.format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .save(path)
        )
        print(f"[OK] Wrote Delta -> {path}")

    spark = (
        SparkSession.builder.master("local[*]")
        .config("spark.hadoop.fs.s3a.endpoint", f"http://{MINIO_ENDPOINT}")
        .config("spark.hadoop.fs.s3a.access.key", MINIO_ACCESS_KEY)
        .config("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET_KEY)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.databricks.delta.schema.autoMerge.enabled", "true")
        .appName("Gold application batching")
        .getOrCreate()
    )

    silver_table = spark.read.format("delta").load(SILVER_PATH_APPLICATION)

    gold_fact = silver_table.select(
        "sk_id_curr",
        "amt_income_total",
        "amt_credit",
        "amt_annuity",
        "amt_goods_price",
        "event_ts"
    )

    gold_dim_user_demographic = (
        silver_table.select(
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

    gold_dim_user_contact = silver_table.select(
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

    gold_dim_user_region = silver_table.select(
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

    gold_dim_asset = silver_table.select(
        "sk_id_curr",
        "flag_own_car",
        "flag_own_realty",
        "own_car_age",
        "name_housing_type",
        "name_type_suite",
        "event_ts"
    )

    gold_dim_user_income = silver_table.select(
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

    gold_dim_external_source = silver_table.select(
        "sk_id_curr",
        "ext_source_1",
        "ext_source_2",
        "ext_source_3",
        "event_ts"
    )

    gold_dim_application_time = silver_table.select(
        "sk_id_curr",
        "days_registration",
        "event_ts"
    ).withColumn(
        "days_id_publish",
        F.to_date(
            F.date_add(F.current_date(), F.col("days_id_publish").cast("int"))
        )
    ).withColumn(
        "is_weekend",
        F.when(F.col("weekday_appr_process_start").isin("SATURDAY", "SUNDAY"), F.lit(1)).otherwise(F.lit(0))
    ).withColumn(
        "is_working_hour",
        F.when(
            F.col("hour_appr_process_start").between(9, 18), F.lit(1)
        ).otherwise(F.lit(0))
    )

    docs_cols = [f"flag_document_{i}" for i in range(2,22)]

    aggregated_cols = [
        col for col in silver_table.columns if col.endswith(("_avg", "_max", "_min", "_medi"))
    ]

    gold_dim_provided_docs = silver_table.select(
        "sk_id_curr",
        *docs_cols,
        "event_ts",
    )

    gold_dim_aggregated = silver_table.select(
        "sk_id_curr",
        *aggregated_cols,
        "event_ts",
    )

    write_delta_overwrite(gold_fact, "fact_loan")
    write_delta_overwrite(gold_dim_user_demographic, "dim_user_demographic")
    write_delta_overwrite(gold_dim_user_contact, "dim_user_contact")
    write_delta_overwrite(gold_dim_user_region, "dim_user_region")
    write_delta_overwrite(gold_dim_asset, "dim_asset")
    write_delta_overwrite(gold_dim_user_income, "dim_user_income")
    write_delta_overwrite(gold_dim_external_source, "dim_external_source")
    write_delta_overwrite(gold_dim_application_time, "dim_application_time")
    write_delta_overwrite(gold_dim_provided_docs, "dim_provided_docs")
    write_delta_overwrite(gold_dim_aggregated, "dim_aggregated")

if __name__ == "__main__":
    main()
