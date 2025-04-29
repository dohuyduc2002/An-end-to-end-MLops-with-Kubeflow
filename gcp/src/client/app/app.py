from typing import List, Dict, Any, Optional
from time import time

import numpy as np
import pandas as pd
from fastapi import FastAPI, Body
from pydantic import BaseModel

import xgboost as xgb

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.prometheus import PrometheusMetricReader, start_http_server

from pathlib import Path
import joblib
from loguru import logger

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = "data"
JOBLIB_DIR = "joblib"

# ======= Load Transformer and Model ========
transformer_path = Path(JOBLIB_DIR) / "transformer.joblib"
# Removed the redundant model_path assignment to "model.xgb"
data_path = Path(DATA_DIR) / "application_test.csv"
model_path = Path(JOBLIB_DIR) / "model.joblib"

if not transformer_path.exists():
    raise FileNotFoundError(f"Transformer file not found at {transformer_path}")
if not model_path.exists():
    raise FileNotFoundError(f"Model file not found at {model_path}")
if not data_path.exists():
    raise FileNotFoundError(f"Data file not found at {data_path}")

# Load transformer
transformer = joblib.load(transformer_path)
# Load model
xgb_model = joblib.load(str(model_path))
logger.info(f"Loaded model from {model_path}")

# ======= OpenTelemetry gauges ========
reader = PrometheusMetricReader()
provider = MeterProvider(metric_readers=[reader])
metrics.set_meter_provider(provider)
meter = metrics.get_meter_provider().get_meter("prediction_api")
start_http_server(port=8001, addr="0.0.0.0")

last_avg_entropy = 0.0
last_avg_confidence = 0.0

meter.create_observable_gauge(
    name="api_prediction_entropy",
    callbacks=[lambda opts: [metrics.Observation(last_avg_entropy)]],
    description="Average prediction entropy",
)
meter.create_observable_gauge(
    name="api_avg_confidence",
    callbacks=[lambda opts: [metrics.Observation(last_avg_confidence)]],
    description="Average model confidence",
)

# ======= FastAPI setup ========
app = FastAPI(root_path="/prediction-api")

class RawItem(BaseModel):
    SK_ID_CURR: Optional[int] = None
    NAME_CONTRACT_TYPE: Optional[str] = None
    CODE_GENDER: Optional[str] = None
    FLAG_OWN_CAR: Optional[str] = None
    FLAG_OWN_REALTY: Optional[str] = None
    CNT_CHILDREN: Optional[int] = None
    AMT_INCOME_TOTAL: Optional[float] = None
    AMT_CREDIT: Optional[float] = None
    AMT_ANNUITY: Optional[float] = None
    AMT_GOODS_PRICE: Optional[float] = None
    NAME_TYPE_SUITE: Optional[str] = None
    NAME_INCOME_TYPE: Optional[str] = None
    NAME_EDUCATION_TYPE: Optional[str] = None
    NAME_FAMILY_STATUS: Optional[str] = None
    NAME_HOUSING_TYPE: Optional[str] = None
    REGION_POPULATION_RELATIVE: Optional[float] = None
    DAYS_BIRTH: Optional[int] = None
    DAYS_EMPLOYED: Optional[float] = None
    DAYS_REGISTRATION: Optional[float] = None
    DAYS_ID_PUBLISH: Optional[int] = None
    OWN_CAR_AGE: Optional[int] = None
    FLAG_MOBIL: Optional[int] = None
    FLAG_EMP_PHONE: Optional[int] = None
    FLAG_WORK_PHONE: Optional[int] = None
    FLAG_CONT_MOBILE: Optional[int] = None
    FLAG_PHONE: Optional[int] = None
    FLAG_EMAIL: Optional[int] = None
    OCCUPATION_TYPE: Optional[str] = None
    CNT_FAM_MEMBERS: Optional[float] = None
    REGION_RATING_CLIENT: Optional[int] = None
    REGION_RATING_CLIENT_W_CITY: Optional[int] = None
    WEEKDAY_APPR_PROCESS_START: Optional[str] = None
    HOUR_APPR_PROCESS_START: Optional[int] = None
    REG_REGION_NOT_LIVE_REGION: Optional[int] = None
    REG_REGION_NOT_WORK_REGION: Optional[int] = None
    LIVE_REGION_NOT_WORK_REGION: Optional[int] = None
    REG_CITY_NOT_LIVE_CITY: Optional[int] = None
    REG_CITY_NOT_WORK_CITY: Optional[int] = None
    LIVE_CITY_NOT_WORK_CITY: Optional[int] = None
    ORGANIZATION_TYPE: Optional[str] = None
    EXT_SOURCE_1: Optional[float] = None
    EXT_SOURCE_2: Optional[float] = None
    EXT_SOURCE_3: Optional[float] = None
    APARTMENTS_AVG: Optional[float] = None
    BASEMENTAREA_AVG: Optional[float] = None
    YEARS_BEGINEXPLUATATION_AVG: Optional[float] = None
    YEARS_BUILD_AVG: Optional[float] = None
    COMMONAREA_AVG: Optional[float] = None
    ELEVATORS_AVG: Optional[float] = None
    ENTRANCES_AVG: Optional[float] = None
    FLOORSMAX_AVG: Optional[float] = None
    FLOORSMIN_AVG: Optional[float] = None
    LANDAREA_AVG: Optional[float] = None
    LIVINGAPARTMENTS_AVG: Optional[float] = None
    LIVINGAREA_AVG: Optional[float] = None
    NONLIVINGAPARTMENTS_AVG: Optional[float] = None
    NONLIVINGAREA_AVG: Optional[float] = None
    APARTMENTS_MODE: Optional[float] = None
    BASEMENTAREA_MODE: Optional[float] = None
    YEARS_BEGINEXPLUATATION_MODE: Optional[float] = None
    YEARS_BUILD_MODE: Optional[float] = None
    COMMONAREA_MODE: Optional[float] = None
    ELEVATORS_MODE: Optional[float] = None
    ENTRANCES_MODE: Optional[float] = None
    FLOORSMAX_MODE: Optional[float] = None
    FLOORSMIN_MODE: Optional[float] = None
    LANDAREA_MODE: Optional[float] = None
    LIVINGAPARTMENTS_MODE: Optional[float] = None
    LIVINGAREA_MODE: Optional[float] = None
    NONLIVINGAPARTMENTS_MODE: Optional[float] = None
    NONLIVINGAREA_MODE: Optional[float] = None
    APARTMENTS_MEDI: Optional[float] = None
    BASEMENTAREA_MEDI: Optional[float] = None
    YEARS_BEGINEXPLUATATION_MEDI: Optional[float] = None
    YEARS_BUILD_MEDI: Optional[float] = None
    COMMONAREA_MEDI: Optional[float] = None
    ELEVATORS_MEDI: Optional[float] = None
    ENTRANCES_MEDI: Optional[float] = None
    FLOORSMAX_MEDI: Optional[float] = None
    FLOORSMIN_MEDI: Optional[float] = None
    LANDAREA_MEDI: Optional[float] = None
    LIVINGAPARTMENTS_MEDI: Optional[float] = None
    LIVINGAREA_MEDI: Optional[float] = None
    NONLIVINGAPARTMENTS_MEDI: Optional[float] = None
    NONLIVINGAREA_MEDI: Optional[float] = None
    FONDKAPREMONT_MODE: Optional[str] = None
    HOUSETYPE_MODE: Optional[float] = None
    TOTALAREA_MODE: Optional[str] = None
    WALLSMATERIAL_MODE: Optional[str] = None
    EMERGENCYSTATE_MODE: Optional[float] = None
    OBS_30_CNT_SOCIAL_CIRCLE: Optional[float] = None
    DEF_30_CNT_SOCIAL_CIRCLE: Optional[float] = None
    OBS_60_CNT_SOCIAL_CIRCLE: Optional[float] = None
    DEF_60_CNT_SOCIAL_CIRCLE: Optional[float] = None
    DAYS_LAST_PHONE_CHANGE: Optional[float] = None
    FLAG_DOCUMENT_2: Optional[int] = None
    FLAG_DOCUMENT_3: Optional[int] = None
    FLAG_DOCUMENT_4: Optional[int] = None
    FLAG_DOCUMENT_5: Optional[int] = None
    FLAG_DOCUMENT_6: Optional[int] = None
    FLAG_DOCUMENT_7: Optional[int] = None
    FLAG_DOCUMENT_8: Optional[int] = None
    FLAG_DOCUMENT_9: Optional[int] = None
    FLAG_DOCUMENT_10: Optional[int] = None
    FLAG_DOCUMENT_11: Optional[int] = None
    FLAG_DOCUMENT_12: Optional[int] = None
    FLAG_DOCUMENT_13: Optional[int] = None
    FLAG_DOCUMENT_14: Optional[int] = None
    FLAG_DOCUMENT_15: Optional[int] = None
    FLAG_DOCUMENT_16: Optional[int] = None
    FLAG_DOCUMENT_17: Optional[int] = None
    FLAG_DOCUMENT_18: Optional[int] = None
    FLAG_DOCUMENT_19: Optional[int] = None
    FLAG_DOCUMENT_20: Optional[int] = None
    FLAG_DOCUMENT_21: Optional[int] = None
    AMT_REQ_CREDIT_BUREAU_HOUR: Optional[float] = None
    AMT_REQ_CREDIT_BUREAU_DAY: Optional[float] = None
    AMT_REQ_CREDIT_BUREAU_WEEK: Optional[float] = None
    AMT_REQ_CREDIT_BUREAU_MON: Optional[float] = None
    AMT_REQ_CREDIT_BUREAU_QRT: Optional[float] = None
    AMT_REQ_CREDIT_BUREAU_YEAR: Optional[float] = None

    class Config:
        extra = "ignore"

def entropy(p: np.ndarray) -> float:
    return float(-np.sum(p * np.log2(p + 1e-10)))

def confidence(p: np.ndarray) -> float:
    return float(p.max())

@app.get("/")
def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.post("/Prediction")
async def predict(items: List[RawItem] = Body(...)) -> Dict[str, Any]:
    global last_avg_entropy, last_avg_confidence
    t0 = time()
    df_raw = pd.DataFrame([i.dict() for i in items]).replace({None: np.nan})

    binning = transformer["binning_process"]
    selector = transformer["selector"]

    df_raw = df_raw[[col for col in df_raw.columns if col in binning.variable_names]]

    # pipeline
    X_binned = binning.transform(df_raw)
    X = selector.transform(X_binned)

    # predict
    proba = xgb_model.predict_proba(X)

    preds = np.argmax(proba, axis=1)

    entropies = [entropy(p) for p in proba]
    confidences = [confidence(p) for p in proba]

    last_avg_entropy = float(np.mean(entropies))
    last_avg_confidence = float(np.mean(confidences))

    return {
        "inference_time_ms": round((time() - t0) * 1000, 2),
        "predictions": [
            {
                "result": "Accept" if y == 0 else "Decline",
                "prob_accept": float(p[0]),
                "prob_decline": float(p[1]),
                "entropy": round(e, 4),
                "confidence": round(c, 4),
            }
            for y, p, e, c in zip(preds, proba, entropies, confidences)
        ],
        "metrics": {
            "avg_entropy": last_avg_entropy,
            "avg_confidence": last_avg_confidence,
        },
    }

@app.post("/Prediction-by-id")
def predict_by_id(id: int) -> Dict[str, Any]:
    global last_avg_entropy, last_avg_confidence
    t0 = time()

    try:
        df_all = pd.read_csv(data_path)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch test data locally: {str(e)}")

    df_row = df_all[df_all["SK_ID_CURR"] == id]
    if df_row.empty:
        return {"error": f"ID {id} not found"}

    binning = transformer["binning_process"]
    selector = transformer["selector"]

    df_row = df_row[[col for col in df_row.columns if col in binning.variable_names]]

    X_binned = binning.transform(df_row)
    X = selector.transform(X_binned)

    proba = xgb_model.predict_proba(X)
    preds = np.argmax(proba, axis=1)

    entropies = [entropy(p) for p in proba]
    confidences = [confidence(p) for p in proba]

    last_avg_entropy = float(np.mean(entropies))
    last_avg_confidence = float(np.mean(confidences))

    return {
        "inference_time_ms": round((time() - t0) * 1000, 2),
        "predictions": [
            {
                "result": "Accept" if y == 0 else "Decline",
                "prob_accept": float(p[0]),
                "prob_decline": float(p[1]),
                "entropy": round(e, 4),
                "confidence": round(c, 4),
            }
            for y, p, e, c in zip(preds, proba, entropies, confidences)
        ],
        "metrics": {
            "avg_entropy": last_avg_entropy,
            "avg_confidence": last_avg_confidence,
        },
    }
