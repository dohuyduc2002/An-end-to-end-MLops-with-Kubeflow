from pyspark.sql import SparkSession
from pyspark.sql import functions as F

import sys
from pathlib import Path

from gold_etl_schema import (
    gold_dim_user_demographic_schema,
    gold_dim_user_contact_schema,
    gold_dim_user_region_schema,
    gold_dim_external_source_schema,
    gold_dim_user_income_schema,
    gold_dim_application_time_schema,
    gold_dim_provided_docs_schema,
    gold_dim_aggregated_schema,
    gold_dim_asset_and_bureau_schema,
    gold_fact_loan_schema,
    gold_fact_bureau_balance_schema
)

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    MINIO_ENDPOINT,
    MINIO_ACCESS_KEY,
    MINIO_SECRET_KEY,
    SILVER_PATH_APPLICATION,
    DATA_MART_GOLD_PATH,
    SILVER_PATH_MERGED,
    write_with_uc_sql,
)

def main():
    spark = (
        SparkSession.builder
        .config("spark.hadoop.fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem")
        .config("spark.hadoop.fs.AbstractFileSystem.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS")
        .config("spark.hadoop.google.cloud.auth.service.account.enable", "true")
        .config("spark.hadoop.google.cloud.auth.service.account.json.keyfile", "/var/secrets/gcp/gcp-key.json")
        .appName("Gold application batching")
        .getOrCreate()
    )
    
    silver_tables = {
        "fact_loan": f"{SILVER_PATH_APPLICATION}/fact_loan",
        "fact_bureau_balance": f"{SILVER_PATH_MERGED}/fact_bureau_balance",
        "dim_bureau": f"{SILVER_PATH_MERGED}/dim_bureau",
        "dim_user_demographic": f"{SILVER_PATH_APPLICATION}/dim_user_demographic",
        "dim_user_contact": f"{SILVER_PATH_APPLICATION}/dim_user_contact",
        "dim_user_region": f"{SILVER_PATH_APPLICATION}/dim_user_region",
        "dim_asset": f"{SILVER_PATH_APPLICATION}/dim_asset",
        "dim_user_income": f"{SILVER_PATH_APPLICATION}/dim_user_income",
        "dim_external_source": f"{SILVER_PATH_APPLICATION}/dim_external_source",
        "dim_application_time": f"{SILVER_PATH_APPLICATION}/dim_application_time",
        "dim_provided_docs": f"{SILVER_PATH_APPLICATION}/dim_provided_docs",
        "dim_aggregated": f"{SILVER_PATH_APPLICATION}/dim_aggregated",
    }
    
    silver_dfs = {
        name: spark.read.format("delta").load(path)
        for name, path in silver_tables.items()
    }
    
    gold_dim_user_demographic = (
        silver_dfs["dim_user_demographic"]
        .filter(F.col("is_current") == True)
        .filter(F.col("end_date") > F.current_timestamp())
        .drop("is_current", "end_date")
    )
    
    gold_dim_user_contact = (
        silver_dfs["dim_user_contact"]
        .filter(F.col("is_current") == True)
        .filter(F.col("end_date") > F.current_timestamp())
        .drop("is_current", "end_date")
    )
    
    gold_dim_user_region = (
        silver_dfs["dim_user_region"]
        .filter(F.col("is_current") == True)
        .filter(F.col("end_date") > F.current_timestamp())
        .drop("is_current", "end_date")
    )
    
    gold_dim_external_source = (
        silver_dfs["dim_external_source"]
        .filter(F.col("is_current") == True)
        .filter(F.col("end_date") > F.current_timestamp())
        .drop("is_current", "end_date")
    )
    
    gold_dim_user_income = (
        silver_dfs["dim_user_income"]
        .filter(F.col("is_current") == True)
        .filter(F.col("end_date") > F.current_timestamp())
        .drop("is_current", "end_date")
    )
    
    gold_dim_aggregated = (
        silver_dfs["dim_aggregated"]
        .filter(F.col("is_current") == True)
        .filter(F.col("end_date") > F.current_timestamp())
        .drop("is_current", "end_date")
    )
    gold_dim_provided_docs = (
        silver_dfs["dim_provided_docs"]
        .filter(F.col("is_current") == True)
        .filter(F.col("end_date") > F.current_timestamp())
        .drop("is_current", "end_date")
    )
    
    gold_dim_application_time = (
        silver_dfs["dim_application_time"]
        .filter(F.col("is_current") == True)
        .filter(F.col("end_date") > F.current_timestamp())
        .drop("is_current", "end_date")
    )
    
    gold_dim_asset_and_bureau = (
    silver_dfs["dim_asset"]
    .filter(F.col("is_current") == True)
    .filter(F.col("end_date") > F.current_timestamp())
    .withColumnRenamed("is_current", "asset_is_current")
    # .withColumnRenamed("effective_date", "asset_effective_date")
    .withColumnRenamed("end_date", "asset_end_date")
    .join(
        silver_dfs["dim_bureau"]
        .filter(F.col("is_current") == True)
        .filter(F.col("end_date") > F.current_timestamp())
        .withColumnRenamed("is_current", "bureau_is_current")
        .withColumnRenamed("effective_date", "bureau_effective_date")
        .withColumnRenamed("end_date", "bureau_end_date"),
        on="sk_id_curr",
        how="left"
    )
    .drop("bureau_is_current", "bureau_effective_date", "bureau_end_date", "asset_is_current", "asset_end_date")
    )
    
    gold_fact_loan = (
        silver_dfs["fact_loan"]
        .filter(F.col("is_current") == True)
        .filter(F.col("end_date") > F.current_timestamp())
        .drop("is_current", "end_date")
    )
    
    gold_fact_bureau_balance = (
        silver_dfs["fact_bureau_balance"]
        .filter(F.col("is_current") == True)
        .filter(F.col("end_date") > F.current_timestamp())
        .drop("is_current", "end_date")
    )
    
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS gold LOCATION 'gs://unity-catalog-dhduc/data-mart'")

    gold_tables = {
        "dim_user_demographic": gold_dim_user_demographic,
        "dim_user_contact": gold_dim_user_contact,
        "dim_user_region": gold_dim_user_region,
        "dim_external_source": gold_dim_external_source,
        "dim_user_income": gold_dim_user_income,
        "dim_aggregated": gold_dim_aggregated,
        "dim_provided_docs": gold_dim_provided_docs,
        "dim_application_time": gold_dim_application_time,
        "dim_asset_and_bureau": gold_dim_asset_and_bureau,
        "fact_loan": gold_fact_loan,
        "fact_bureau_balance": gold_fact_bureau_balance
    }
    
    gold_tables_schemas = {
        "dim_user_demographic": gold_dim_user_demographic_schema,
        "dim_user_contact": gold_dim_user_contact_schema,
        "dim_user_region": gold_dim_user_region_schema,
        "dim_external_source": gold_dim_external_source_schema,
        "dim_user_income": gold_dim_user_income_schema,
        "dim_aggregated": gold_dim_aggregated_schema,
        "dim_provided_docs": gold_dim_provided_docs_schema,
        "dim_application_time": gold_dim_application_time_schema,
        "dim_asset_and_bureau": gold_dim_asset_and_bureau_schema,
        "fact_loan": gold_fact_loan_schema,
        "fact_bureau_balance": gold_fact_bureau_balance_schema
    }
    
    # --- Gold tables ---
    for name, df in gold_tables.items():
        schema = gold_tables_schemas[name]

        write_with_uc_sql(
            spark,
            df,
            layer="gold",
            table_name=name,
            schema=schema,
            base_path=DATA_MART_GOLD_PATH,
            mode="overwrite"   # gold layer thường build lại từ đầu => overwrite
        )


    
if __name__ == "__main__":
    main()