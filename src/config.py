import logging
import os
from pathlib import Path
from dotenv import load_dotenv

# Locate the .env file at the project root
BASE_DIR = Path(__file__).resolve().parents[1]  # Go up two levels from src/log
ENV_PATH = BASE_DIR / ".env"

# Load environment variables
load_dotenv(dotenv_path=ENV_PATH)

# Logging
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

# Config
class Config:
    MLFLOW_URI = os.environ.get("MLFLOW_URI")
    PREDICTOR_API_PORT = os.environ.get("PREDICTOR_API_PORT") 
    SERVICE_NAME_TELEMETRY = os.environ.get("SERVICE_NAME")
