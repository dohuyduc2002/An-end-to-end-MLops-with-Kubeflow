import os
from time import time
from typing import List

import numpy as np
from fastapi import FastAPI, Body
import mlflow
from mlflow.tracking import MlflowClient
from pydantic import BaseModel
from config import logging, Config  # import logging and configuration

# Set up MLflow tracking using configuration
mlflow.set_tracking_uri(Config.MLFLOW_URI)
client = MlflowClient()

# Specify the model name registered in MLflow
model_name = "v1_XGBoost"

# Retrieve the latest model version from the 'Production' stage.
latest_model_versions = client.get_latest_versions(model_name, stages=["Production"])
if not latest_model_versions:
    # If no production model is found, fall back to the default stage.
    latest_model_versions = client.get_latest_versions(model_name, stages=["None"])
model_version = latest_model_versions[0].version
model_uri = f"models:/{model_name}/{model_version}"
logging.info(f"Loading model from MLflow registry: {model_uri}")

# Load the model from the MLflow registry.
model = mlflow.pyfunc.load_model(model_uri)

app = FastAPI()

# Define the order for feature extraction.
FEATURES = [
    "CODE_GENDER",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "FLAG_EMP_PHONE",
    "REG_CITY_NOT_LIVE_CITY",
    "REG_CITY_NOT_WORK_CITY",
    "FLAG_DOCUMENT_3"
]

# Model for individual data records.
class DataItem(BaseModel):
    CODE_GENDER: float
    NAME_INCOME_TYPE: float
    NAME_EDUCATION_TYPE: float
    NAME_FAMILY_STATUS: float
    DAYS_BIRTH: float
    DAYS_EMPLOYED: float
    FLAG_EMP_PHONE: float
    REG_CITY_NOT_LIVE_CITY: float
    REG_CITY_NOT_WORK_CITY: float
    FLAG_DOCUMENT_3: float

@app.get("/")
def home():
    return "API is working."

# Update the endpoint to accept a raw list of DataItem objects.
@app.post("/Prediction")
async def underwrite_predict(
    input: List[DataItem] = Body(
        ...,
        example=[
            {
                "CODE_GENDER": 0.15432825417103907,
                "NAME_INCOME_TYPE": -0.18921347783967535,
                "NAME_EDUCATION_TYPE": 0.4411157211533453,
                "NAME_FAMILY_STATUS": 0.0712219459830889,
                "DAYS_BIRTH": 0.20261567037598555,
                "DAYS_EMPLOYED": 0.0007483049659318368,
                "FLAG_EMP_PHONE": -0.07661019233668907,
                "REG_CITY_NOT_LIVE_CITY": 0.048430407255087446,
                "REG_CITY_NOT_WORK_CITY": 0.10713747002236401,
                "FLAG_DOCUMENT_3": -0.09976667709096392
            },
            {
                "CODE_GENDER": -0.2509313514955317,
                "NAME_INCOME_TYPE": -0.18921347783967535,
                "NAME_EDUCATION_TYPE": -0.11303014010020201,
                "NAME_FAMILY_STATUS": 0.0712219459830889,
                "DAYS_BIRTH": 0.13254604970005926,
                "DAYS_EMPLOYED": 0.3914664457865674,
                "FLAG_EMP_PHONE": -0.07661019233668907,
                "REG_CITY_NOT_LIVE_CITY": 0.048430407255087446,
                "REG_CITY_NOT_WORK_CITY": 0.10713747002236401,
                "FLAG_DOCUMENT_3": -0.09976667709096392
            }
        ]
    )
):
    start_time = time()
    
    # Extract the features in the correct order from each record.
    data_list = []
    for item in input:
        item_dict = item.dict()
        row = [item_dict[feature] for feature in FEATURES]
        data_list.append(row)
    x = np.array(data_list)

    logging.info("Making predictions...")
    preds = model.predict(x)

    # Prepare the list of prediction results.
    results = [{"Underwrite result": "Accept to underwrite" if p == 0 else "Decline to underwrite"} for p in preds]

    elapsed_time = time() - start_time
    logging.info(f"Elapsed time: {elapsed_time}")
    
    return results
# uvicorn api:app --host 0.0.0.0 --port 8000
