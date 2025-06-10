import os
from typing import Optional, Tuple
import joblib
import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from mlflow.tracking import MlflowClient
from minio import Minio
from io import BytesIO

from evidently import Dataset, DataDefinition
from evidently import Report
from evidently.metrics import (
    DatasetMissingValueCount,
    EmptyRowsCount,
    DuplicatedColumnsCount,
    CategoryCount,
)


from evidently.presets import DataDriftPreset, ValueStats

load_dotenv(
    override=False
)  # just for local testing, in production, the env is define in Docker image and Kubernetes

# We create a Config class to hold all the API configuration parameters
class ApiConfig:
    def __init__(self):
        self.s3_endpoint: str = os.getenv("S3_ENDPOINT")
        self.s3_access_key: str = os.getenv("S3_ACCESS_KEY")
        self.s3_secret_key: str = os.getenv("S3_SECRET_KEY")
        self.evidently_workspace: str = os.getenv("EVIDENTLY_WORKSPACE")
        self.mlflow_endpoint: str = os.getenv("MLFLOW_ENDPOINT")
        self.model_name: str = os.getenv("MODEL_NAME")
        self.model_type: str = os.getenv("MODEL_TYPE").lower()  # xgb | lgbm
        self.parent_run_id: str = os.getenv("PARENT_RUN_ID")
        self.transformer_artifact_path: str = os.getenv("TRANSFORMER_ARTIFACT_PATH")

    def configure_mlflow(self):
        os.environ["AWS_ACCESS_KEY_ID"] = self.s3_access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = self.s3_secret_key
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://{self.s3_endpoint}"
        mlflow.set_tracking_uri(self.mlflow_endpoint)

    def get_minio_client(self):
        return Minio(
            self.s3_endpoint,
            access_key=self.s3_access_key,
            secret_key=self.s3_secret_key,
            secure=False,
        )


# Create a Predictor class to handle model loading and inference, this class will be used in both POST method 
class Predictor:
    def __init__(self, cfg: ApiConfig):
        self.cfg = cfg
        self.transformer: Optional[dict] = None
        self.model: Optional[object] = None

        self.cfg.configure_mlflow()

    # Artifact loader
    def load_artifacts(self):
        client = MlflowClient()
        # Load transformer feature artifact from mlfow at the parent run
        downloaded_path = client.download_artifacts(
            run_id=self.cfg.parent_run_id,
            path=self.cfg.transformer_artifact_path,
            dst_path="/tmp",
        )
        self.transformer = joblib.load(downloaded_path)

        # Load model from MLflow with the registered stage, in this case we prioritize "Production" stage, the None stage is used for testing
        versions = client.get_latest_versions(
            self.cfg.model_name, stages=["Production"]
        )
        if not versions:
            versions = client.get_latest_versions(self.cfg.model_name, stages=["None"])
        if not versions:
            raise RuntimeError(f"Model '{self.cfg.model_name}' not found in MLflow.")

        # In my case, I have 2 model type, one is XGB and LGBM, each has different class in mlflow so we need to load it accordingly
        model_uri = f"models:/{self.cfg.model_name}/{versions[0].version}"
        if self.cfg.model_type == "xgb":
            self.model = mlflow.xgboost.load_model(model_uri)
        elif self.cfg.model_type == "lgbm":
            self.model = mlflow.lightgbm.load_model(model_uri)
        else:
            raise ValueError(f"Unsupported model type: {self.cfg.model_type}")

    # Inference
    def inference(self, df: pd.DataFrame):
        # The transformer serialization joblib is a dict with 2 separate step:
        binning = self.transformer["opt_binning_process"]
        selector = self.transformer["selector"]
        df = df[[col for col in df.columns if col in binning.variable_names]]

        X = selector.transform(binning.transform(df))
        proba = self.model.predict_proba(X)
        preds = proba.argmax(axis=1)
        return preds, proba


def map_evidently_data(config):
    # Refer to evidently docs for create a Evidently compatiable Dataset:
    # https://docs.evidentlyai.com/docs/library/data_definition
    def get_lists(df):
        numeric = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        category = df.select_dtypes(include=["object"]).columns.tolist()
        for column in ("SK_ID_CURR", "TARGET"):
            if column in numeric:
                numeric.remove(column)
            if column in category:
                category.remove(column)
        return category, numeric

    minio_client = config.get_minio_client()
    train_data = minio_client.get_object("sample-data", "data/application_train.csv")
    test_data = minio_client.get_object("sample-data", "data/application_test.csv")

    # We can only read Minio Object as bytes, so we need to convert it to pandas DataFrame
    # https://stackoverflow.com/questions/55223401/minio-python-client-upload-bytes-directly
    df_train = pd.read_csv(BytesIO(train_data.read()))
    df_test = pd.read_csv(BytesIO(test_data.read()))

    categorical_cols, numerical_cols = get_lists(df_train)

    definition = DataDefinition(
        numerical_columns=numerical_cols,
        categorical_columns=categorical_cols,
    )

    reference_data = Dataset.from_pandas(df_train, data_definition=definition)
    current_data = Dataset.from_pandas(df_test, data_definition=definition)

    return reference_data, current_data


def custom_evidently_report(reference_data, current_data):
    # After create a Evidently Dataset, we can create a Report with the metrics we want to calculate
    # https://docs.evidentlyai.com/docs/library/tests
    # In this case, I customize the report with some metrics that I want to calculate 
    metrics = [
        DatasetMissingValueCount(),
        DuplicatedColumnsCount(),
        EmptyRowsCount(),
        DataDriftPreset(),
        ValueStats(column="NAME_CONTRACT_TYPE"),
        CategoryCount(
            column="NAME_FAMILY_STATUS",
            categories=[
                "Married",
                "Single / not married",
                "Civil marriage",
                "Separated",
                "Widow",
                "Unknown",
            ],
        ),
    ]
    # The Report class is used to run the metrics on the reference and current data
    report = Report(metrics=metrics)
    # MUST return the snapshot object to use it to add to workspace later
    snapshot = report.run(reference_data, current_data)
    return snapshot
