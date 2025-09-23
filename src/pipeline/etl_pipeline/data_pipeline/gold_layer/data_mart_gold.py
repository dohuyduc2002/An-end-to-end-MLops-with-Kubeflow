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
    DATA_MART_GOLD_PATH,
    write_with_delta_sql,
)

def main():
    spark = (
        SparkSession.builder
        .appName("Data Mart Gold Layer")
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

    silver_layer_schema = "homecredit_silver"
    silver_tables = {
        "fact_loan": f"{silver_layer_schema}.fact_loan",
        "fact_bureau_balance": f"{silver_layer_schema}.fact_bureau_balance",
        "dim_bureau": f"{silver_layer_schema}.dim_bureau",
        "dim_user_demographic": f"{silver_layer_schema}.dim_user_demographic",
        "dim_user_contact": f"{silver_layer_schema}.dim_user_contact",
        "dim_user_region": f"{silver_layer_schema}.dim_user_region",
        "dim_asset": f"{silver_layer_schema}.dim_asset",
        "dim_user_income": f"{silver_layer_schema}.dim_user_income",
        "dim_external_source": f"{silver_layer_schema}.dim_external_source",
        "dim_application_time": f"{silver_layer_schema}.dim_application_time",
        "dim_provided_docs": f"{silver_layer_schema}.dim_provided_docs",
        "dim_aggregated": f"{silver_layer_schema}.dim_aggregated",
    }
    
    silver_dfs = {
        name: spark.read.table(path) for name, path in silver_tables.items()
    }
    
    gold_dim_user_demographic = (
        silver_dfs["dim_user_demographic"]
        .filter(F.col("is_current") == True)
        .filter(F.col("end_date") > F.current_timestamp())
        .drop("is_current", "end_date","event_ts")
    )
    
    gold_dim_user_contact = (
        silver_dfs["dim_user_contact"]
        .filter(F.col("is_current") == True)
        .filter(F.col("end_date") > F.current_timestamp())
        .drop("is_current", "end_date","event_ts")
    )
    
    gold_dim_user_region = (
        silver_dfs["dim_user_region"]
        .filter(F.col("is_current") == True)
        .filter(F.col("end_date") > F.current_timestamp())
        .drop("is_current", "end_date","event_ts")
    )
    
    gold_dim_external_source = (
        silver_dfs["dim_external_source"]
        .filter(F.col("is_current") == True)
        .filter(F.col("end_date") > F.current_timestamp())
        .drop("is_current", "end_date","event_ts")
    )
    
    gold_dim_user_income = (
        silver_dfs["dim_user_income"]
        .filter(F.col("is_current") == True)
        .filter(F.col("end_date") > F.current_timestamp())
        .drop("is_current", "end_date","event_ts")
    )
    
    gold_dim_aggregated = (
        silver_dfs["dim_aggregated"]
        .filter(F.col("is_current") == True)
        .filter(F.col("end_date") > F.current_timestamp())
        .drop("is_current", "end_date","event_ts")
    )
    gold_dim_provided_docs = (
        silver_dfs["dim_provided_docs"]
        .filter(F.col("is_current") == True)
        .filter(F.col("end_date") > F.current_timestamp())
        .drop("is_current", "end_date","event_ts")
    )
    
    gold_dim_application_time = (
        silver_dfs["dim_application_time"]
        .filter(F.col("is_current") == True)
        .filter(F.col("end_date") > F.current_timestamp())
        .drop("is_current", "end_date","event_ts")
    )
    
    gold_dim_asset_and_bureau = (
    silver_dfs["dim_asset"]
    .filter(F.col("is_current") == True)
    .filter(F.col("end_date") > F.current_timestamp())
    .withColumnRenamed("is_current", "asset_is_current")
    # .withColumnRenamed("effective_date", "asset_effective_date")
    .withColumnRenamed("end_date", "asset_end_date")
    .withColumnRenamed("event_ts", "event_ts_asset")
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
    .drop("bureau_is_current", "bureau_effective_date", "bureau_end_date", "asset_is_current", "asset_end_date"," event_ts","event_ts_asset")
    )
    
    gold_fact_loan = (
        silver_dfs["fact_loan"]
        .filter(F.col("event_ts") == F.current_timestamp())
    )
    
    gold_fact_bureau_balance = (
        silver_dfs["fact_bureau_balance"]
        .filter(F.col("event_ts") == F.current_timestamp())
    )
    
    spark.sql("CREATE SCHEMA IF NOT EXISTS homecredit_gold")

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
    
    #  Gold tables 
    for name, df in gold_tables.items():
        schema = gold_tables_schemas[name]

        write_with_delta_sql(
            spark,
            df,
            table_name=name,
            schema="homecredit_gold",
            base_path=DATA_MART_GOLD_PATH,
            table_ddl=schema,
            mode="overwrite"
        )


    
if __name__ == "__main__":
    main()