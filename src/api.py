import os
import re
import mlflow
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
from src.config import Config, logging

# Set MLflow tracking URI from config
mlflow.set_tracking_uri(Config.MLFLOW_URI)

############################################
# Utility Function: load_latest_model_and_mappings
############################################
def load_latest_model_and_mappings(model_name: str):
    """Load the latest model version and its schema mappings using regex to identify the schema file."""
    client = MlflowClient()

    try:
        # Get the latest model version using an alias (e.g., "current")
        logging.info(f"Attempting to load latest version of model: {model_name}")
        latest_version = client.get_model_version_by_alias(model_name, "current")
        run_id = latest_version.run_id
        logging.info(f"Found model version: {latest_version.version} with run_id: {run_id}")

        if not run_id:
            logging.error("No run_id found for the model version")
            return None, None

        # Load the model using the MLflow model registry alias syntax
        model = mlflow.pyfunc.load_model(f"models:/{model_name}@current")
        logging.info("Model loaded successfully")

        # List artifacts for the run to search for a schema mappings file
        try:
            logging.info(f"Listing artifacts for run_id: {run_id}")
            artifacts = client.list_artifacts(run_id)
            artifact_paths = [art.path for art in artifacts]
            logging.info(f"Available artifacts: {artifact_paths}")

            # Define a regex pattern to match file names ending with '_schema_mappings.json'
            schema_pattern = re.compile(r'.*_schema_mappings\.json')
            schema_file = None

            for art in artifact_paths:
                if schema_pattern.match(art):
                    schema_file = art
                    logging.info(f"Found schema mappings file: {schema_file}")
                    break

            if schema_file is None:
                logging.error("No schema mappings file found matching the expected pattern")
                return model, None

            # Load the schema mappings file using MLflow artifacts API
            schema_mappings = mlflow.artifacts.load_dict(f"runs:/{run_id}/{schema_file}")
            logging.info("Schema mappings loaded successfully")
            logging.debug(f"Mappings content: {schema_mappings}")

            return model, schema_mappings

        except Exception as e:
            logging.error(f"Error loading schema mappings: {e}")
            return model, None

    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None, None

############################################
# Global Model and Schema Mapping Load
############################################
# Set the model name, using an environment variable (or defaulting to 'my_model')
MODEL_NAME = os.environ.get("MODEL_NAME", "my_model")
model, schema_mappings = load_latest_model_and_mappings(MODEL_NAME)
if model is None:
    logging.error("Model could not be loaded during startup. Check your model registry and alias settings.")

############################################
# FastAPI Application Setup
############################################
app = FastAPI(title="Underwriting Model Inference API")

############################################
# Request Schema
############################################
class PredictionRequest(BaseModel):
    # Accept a list of floats representing feature values.
    # Adjust the expected number of features to match your model's requirements.
    data: List[float]

    class Config:
        schema_extra = {
            "example": {
                "data": [0.5, 1.2, 3.4, 2.1, 0.0, 5.6, 7.8, 1.9]
            }
        }

############################################
# Endpoints
############################################
@app.get("/")
def home():
    return {"message": "Underwriting Model Inference API is up and running!"}

@app.post("/predict")
def predict(request: PredictionRequest):
    """
    Use the loaded MLflow model to predict based on input features.
    Optionally returns the schema mappings as part of the response.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not available for predictions.")

    try:
        # Convert incoming data into the format expected by the model
        input_data = np.array(request.data).reshape(1, -1)
        preds = model.predict(input_data)
        # Depending on the model type, further processing may be required
        prediction = preds[0]

        response = {
            "prediction": prediction,
            "schema_mappings": schema_mappings  # optionally return for debugging/reference
        }
        return response

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during prediction.")

############################################
# To run the API:
# uvicorn api:app --host 0.0.0.0 --port <Config.PREDICTOR_API_PORT or 8000>
############################################
if __name__ == "__main__":
    import uvicorn
    port = int(Config.PREDICTOR_API_PORT) if Config.PREDICTOR_API_PORT is not None else 8000
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)
