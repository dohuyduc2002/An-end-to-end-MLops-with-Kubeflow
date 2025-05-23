import os
import asyncio
from typing import Optional, Tuple

import joblib
import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

load_dotenv(override=False)


# 1) Configuration
class ApiConfig:
    def __init__(self) -> None:
        self.s3_endpoint: str = os.getenv("S3_ENDPOINT", "")
        self.s3_access_key: str = os.getenv("S3_ACCESS_KEY", "")
        self.s3_secret_key: str = os.getenv("S3_SECRET_KEY", "")
        self.mlflow_endpoint: str = os.getenv("MLFLOW_ENDPOINT", "")
        self.model_name: str = os.getenv("MODEL_NAME", "")
        self.model_type: str = os.getenv("MODEL_TYPE", "").lower()  # xgb | lgbm
        self.parent_run_id: str = os.getenv("PARENT_RUN_ID", "")
        self.transformer_artifact_path: str = os.getenv("TRANSFORMER_ARTIFACT_PATH", "")

    def configure_mlflow(self) -> None:
        os.environ["AWS_ACCESS_KEY_ID"] = self.s3_access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = self.s3_secret_key
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://{self.s3_endpoint}"
        mlflow.set_tracking_uri(self.mlflow_endpoint)

    def get_minio_client(self):
        from minio import Minio

        return Minio(
            self.s3_endpoint,
            access_key=self.s3_access_key,
            secret_key=self.s3_secret_key,
            secure=False,
        )


class Predictor:

    def __init__(self, cfg: ApiConfig) -> None:
        self.cfg = cfg
        self.transformer: Optional[dict] = None
        self.model: Optional[object] = None

        self.cfg.configure_mlflow()

    # Artifact loader
    async def load_artifacts_async(self) -> None:

        def _sync_load():
            client = MlflowClient()

            downloaded_path = client.download_artifacts(
                run_id=self.cfg.parent_run_id,
                path=self.cfg.transformer_artifact_path,  
                dst_path="/tmp",
            )
            self.transformer = joblib.load(downloaded_path)

            versions = client.get_latest_versions(
                self.cfg.model_name, stages=["Production"]
            )
            if not versions:
                versions = client.get_latest_versions(
                    self.cfg.model_name, stages=["None"]
                )
            if not versions:
                raise RuntimeError(
                    f"Model '{self.cfg.model_name}' not found in MLflow."
                )

            model_uri = f"models:/{self.cfg.model_name}/{versions[0].version}"
            if self.cfg.model_type == "xgb":
                self.model = mlflow.xgboost.load_model(model_uri)
            elif self.cfg.model_type == "lgbm":
                self.model = mlflow.lightgbm.load_model(model_uri)
            else:
                raise ValueError(f"Unsupported model type: {self.cfg.model_type}")

        await asyncio.to_thread(_sync_load)

    # Inference
    def inference(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        binning = self.transformer["opt_binning_process"]
        selector = self.transformer["selector"]
        df = df[[c for c in df.columns if c in binning.variable_names]]

        X = selector.transform(binning.transform(df))
        proba = self.model.predict_proba(X)
        preds = proba.argmax(axis=1)
        return preds, proba
