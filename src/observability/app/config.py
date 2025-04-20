import os, logging
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[1]  # Go up two levels from src/log
ENV_PATH = BASE_DIR / ".env"
class Config:
    MLFLOW_URI = os.getenv("MLFLOW_URI")

# reuse standard python logging
logging.basicConfig(level=logging.INFO)
