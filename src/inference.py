import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

from src.config import logging, Config


# Set MLflow tracking URI from your environment or configuration
mlflow.set_tracking_uri("http://localhost:5001")


def load_latest_model_and_mappings(model_name: str):
    """Load the latest model version and its category mappings."""
    client = MlflowClient()
    try:
        # Get the latest model version using alias "current"
        logging.info("Attempting to load latest version of model: {}", model_name)
        latest_version = client.get_model_version_by_alias(model_name, "current")
        run_id = latest_version.run_id
        logging.info("Found model version: {} with run_id: {}", latest_version.version, run_id)

        if not run_id:
            logging.error("No run_id found for the model version")
            return None, None

        # Load the model using the model registry alias
        model = mlflow.pyfunc.load_model(f"models:/{model_name}@current")
        logging.info("Model loaded successfully")

        # List artifacts for debugging purposes
        logging.info("Listing artifacts for run_id: {}", run_id)
        artifacts = client.list_artifacts(run_id)
        logging.info("Available artifacts: {}", [art.path for art in artifacts])

        # Load category mappings artifact (assumes file is named "category_mappings.json")
        try:
            category_mappings = mlflow.artifacts.load_dict(f"runs:/{run_id}/category_mappings.json")
            logging.info("Category mappings loaded successfully")
            logging.debug("Mappings content: {}", category_mappings)
            return model, category_mappings
        except Exception as e:
            logging.error("Error loading category mappings: {}", e)
            return None, None

    except Exception as e:
        logging.error("Error loading model: {}", e)
        return None, None


if __name__ == "__main__":
    # Define your model name registered in MLflow
    model_name = "purchase_prediction_model"

    # Load the model and its category mappings
    model, category_mappings = load_latest_model_and_mappings(model_name)

    if model is None or category_mappings is None:
        logging.error("Failed to load model or mappings")
    else:
        # Prepare inference data as a list of dictionaries
        data = [
            {
                "brand": "sumsung",
                "price": 130.76,
                "event_weekday": 2,
                "category_code_level1": "electronics",
                "category_code_level2": "smartphone",
                "activity_count": 1,
            },
            {
                "brand": "video",
                "price": 130.76,
                "event_weekday": 2,
                "category_code_level1": "electronics",
                "category_code_level2": "smartphone",
                "activity_count": 1,
            },
        ]
        # Convert to DataFrame
        df = pd.DataFrame(data)
        logging.info("Input data shape: {}", df.shape)

        # Encode categorical columns using the saved mappings,
        # setting unseen categories to -1
        for col in ["brand", "category_code_level1", "category_code_level2"]:
            mapping = category_mappings.get(col, {})
            df[col] = df[col].map(mapping).fillna(-1)
            logging.info("Encoded column {}", col)

        # Make predictions using the loaded model
        predictions = model.predict(df)
        logging.info("Predictions: {}", predictions)
        logging.info("Encoded Features:\n{}", df.to_string())
