from pyspark.sql import SparkSession, Window
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import (
    col, to_timestamp, to_utc_timestamp
)
from pyspark.sql.functions import (
    max as spark_max,
    min as spark_min,
    lit, row_number, concat, current_timestamp, count, avg, stddev, datediff
)

MINIO_BUCKET = "mlpipeline"
MINIO_ENDPOINT="http://minio-service.kubeflow.svc.cluster.local:9000"
FEATURE_REPO_DIR = "s3a://" + MINIO_BUCKET + "/artifacts/feature_repo/"
DATA_DIR = FEATURE_REPO_DIR + "data/"
INPUT_DIR = DATA_DIR + "input/"
OUTPUT_DIR = DATA_DIR + "output/"


def get_spark():
    return (
        SparkSession
        .builder
        .appName("FeatureEngineeringSpark")
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.1,com.amazonaws:aws-java-sdk-bundle:1.12.458")
        .config("spark.hadoop.fs.s3a.endpoint", MINIO_ENDPOINT)
        .config("spark.hadoop.fs.s3a.access.key", "minio")
        .config("spark.hadoop.fs.s3a.secret.key", "minio123")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .getOrCreate()
    )


def get_features() -> DataFrame:
    spark = get_spark()

    train_set = spark.read.csv(INPUT_DIR + "train.csv", header=True, inferSchema=True)
    test_set = spark.read.csv(INPUT_DIR + "test.csv", header=True, inferSchema=True)
    validate_set = spark.read.csv(INPUT_DIR + "validate.csv", header=True, inferSchema=True)
    train_set = train_set.withColumn("set", lit("train"))
    test_set = test_set.withColumn("set", lit("test"))
    validate_set = validate_set.withColumn("set", lit("valid"))

    all_sets = train_set.unionByName(test_set).unionByName(validate_set)

    all_sets = all_sets.withColumn("fraud", col("fraud") == 1.0)
    all_sets = all_sets.withColumn("repeat_retailer", col("repeat_retailer") == 1.0)
    all_sets = all_sets.withColumn("used_chip", col("used_chip") == 1.0)
    all_sets = all_sets.withColumn("used_pin_number", col("used_pin_number") == 1.0)
    all_sets = all_sets.withColumn("online_order", col("online_order") == 1.0)

    w = Window.orderBy(lit(1))

    all_sets = (
        all_sets
        .withColumn("idx", row_number().over(w))
        .withColumn("user_id", concat(lit("user_"), col("idx") - lit(1)))
        .withColumn("transaction_id", concat(lit("txn_"), col("idx") - lit(1)))
        .drop("idx")
    )

    for date_col in ["created", "updated"]:
        all_sets = all_sets.withColumn(date_col, current_timestamp())

    columns = [
        "user_id",
        "fraud",
        "created",
        "updated",
        "set",
        "distance_from_home",
        "distance_from_last_transaction",
        "ratio_to_median_purchase_price",
    ]

    label_dataset = all_sets.select(*columns)

    user_purchase_history = spark.read.csv(
        INPUT_DIR + "raw_transaction_datasource.csv", header=True,
        inferSchema=True)

    features_df = calculate_point_in_time_features(label_dataset, user_purchase_history)

    selected_columns = all_sets.select(
        "user_id", "created", "used_chip", "used_pin_number", "online_order"
    )

    features_df = features_df.join(
        selected_columns,
        on=["user_id", "created"],
        how="inner"
    )

    return features_df


def calculate_point_in_time_features(label_dataset: DataFrame, transactions_df: DataFrame) -> DataFrame:
    label = (label_dataset
             .withColumn("created_ts", to_timestamp("created", "yyyy-MM-dd HH:mm:ss.SSSSSS"))
             .select("user_id", "created_ts"))
    txn = (transactions_df
           .withColumn("txn_ts", to_timestamp(col("date_of_transaction"), "yyyy-MM-dd HH:mm:ss.SSSSSS"))
           .select("user_id", "txn_ts", "transaction_amount"))

    transactions_before = (label.join(txn, on="user_id", how="inner")
                           .filter(col("txn_ts") < col("created_ts")))

    transactions_before = transactions_before.withColumn(
        "days_between_transactions",
        datediff(col("created_ts"), col("txn_ts"))
    )
    features_df = (transactions_before
                   .groupBy("user_id", "created_ts")
                   .agg(
        count("transaction_amount").alias("num_prev_transactions"),
        avg("transaction_amount").alias("avg_prev_transaction_amount"),
        spark_max("transaction_amount").alias("max_prev_transaction_amount"),
        stddev("transaction_amount").alias("stdv_prev_transaction_amount"),
        spark_min("days_between_transactions").alias("days_since_last_transaction"),
        spark_max("days_between_transactions").alias("days_since_first_transaction"),
    )
                   .na.fill(0)
                   .withColumnRenamed("created_ts", "created")
                   )

    features_df.show(5, truncate=False)

    final = label_dataset.join(
        features_df,
        on=["user_id", "created"],
        how="inner"
    )
    final.show(5, truncate=False)
    return final


def main():
    features = get_features()
    entity_df: DataFrame = features.select("created", "updated", "user_id")
    entity_df = (
        entity_df
        .withColumnRenamed("created", "created_timestamp")
        .withColumnRenamed("updated", "event_timestamp")
    )
    entity_file_name = OUTPUT_DIR + "entity.csv"
    entity_df.write.option("header", True).mode("overwrite").csv(entity_file_name)

    df = features
    df = (
        df.withColumn("created", to_timestamp(col("created"), "yyyy-MM-dd HH:mm:ss.SSSSSS"))
        .withColumn("created", to_utc_timestamp(col("created"), "UTC"))
    )
    df = (
        df.withColumn("updated", to_timestamp(col("updated"), "yyyy-MM-dd HH:mm:ss.SSSSSS"))
        .withColumn("updated", to_utc_timestamp(col("updated"), "UTC"))
    )

    parquet_file_name = OUTPUT_DIR + "features.parquet"
    df.write.mode("overwrite").parquet(parquet_file_name)


if __name__ == "__main__":
    main()