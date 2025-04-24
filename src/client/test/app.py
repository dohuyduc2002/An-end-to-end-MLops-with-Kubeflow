from typing import List, Dict, Any, Optional
from time import time

import numpy as np
import pandas as pd
from fastapi import FastAPI, Body
from pydantic import BaseModel

import joblib
from pathlib import Path
from loguru import logger

# ========== Load local model & transformer ================================
base_path = Path(__file__).resolve().parents[1] / "joblib"
transformer = joblib.load(base_path / "transformer.joblib")
model = joblib.load(base_path / "model.joblib")
logger.info("Loaded transformer and model from local joblib files")

# ========== FastAPI App ===================================================
app = FastAPI()

# ========== Pydantic Model ================================================
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

    class Config:
        extra = "ignore"

# ========== Helper Functions ==============================================
def entropy(p: np.ndarray) -> float:
    return float(-np.sum(p * np.log2(p + 1e-10)))

def confidence(p: np.ndarray) -> float:
    return float(p.max())

# ========== API Endpoints ================================================
@app.get("/")
def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.post("/Prediction")
def predict(items: List[RawItem] = Body(...)) -> Dict[str, Any]:
    t0 = time()

    df_raw = pd.DataFrame([i.dict() for i in items]).replace({None: np.nan})
    binning = transformer["binning_process"]
    selector = transformer["selector"]

    df_raw = df_raw[[col for col in df_raw.columns if col in binning.variable_names]]
    X_binned = binning.transform(df_raw)
    X = selector.transform(X_binned)

    proba = model.predict_proba(X)
    preds = np.argmax(proba, axis=1)

    entropies = [entropy(p) for p in proba]
    confidences = [confidence(p) for p in proba]

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
            "avg_entropy": float(np.mean(entropies)),
            "avg_confidence": float(np.mean(confidences)),
        },
    }
