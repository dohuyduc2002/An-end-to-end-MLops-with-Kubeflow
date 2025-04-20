# monitor.py
import os
import time
import pandas as pd
import numpy as np

from minio import Minio
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from sklearn.metrics import roc_auc_score
from optbinning import BinningProcess
from sklearn.feature_selection import SelectKBest, f_classif

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.prometheus import PrometheusMetricReader, start_http_server

# ─── Telemetry setup ───────────────────────────────────────────────────────
reader   = PrometheusMetricReader()
provider = MeterProvider(metric_readers=[reader])
metrics.set_meter_provider(provider)
meter    = metrics.get_meter_provider().get_meter("drift_monitor")
start_http_server(port=8001, addr="0.0.0.0")

drift_gauge = meter.create_observable_gauge("app_data_drift_score", "Drift 0–1")
gini_gauge  = meter.create_observable_gauge("app_model_gini",     "Gini 0–1")

# ─── Helper: fetch CSV from MinIO ──────────────────────────────────────────
def fetch_from_minio(bucket, obj_name, dest_path):
    client = Minio(
        os.getenv("MINIO_ENDPOINT"),
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
        secure=False
    )
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    client.fget_object(bucket, obj_name, dest_path)
    return pd.read_csv(dest_path)

# ─── Load your train/test frames ──────────────────────────────────────────
# Try ConfigMap paths first; if missing, fall back to MinIO
train_path = "/app/data/application_train.csv"
test_path  = "/app/data/application_test.csv"

if os.path.exists(train_path) and os.path.exists(test_path):
    raw_train = pd.read_csv(train_path)
    raw_test  = pd.read_csv(test_path)
else:
    bucket = os.getenv("MINIO_BUCKET_NAME")
    raw_train = fetch_from_minio(bucket, "data/data/application_train.csv",
                                 "/tmp/application_train.csv")
    raw_test  = fetch_from_minio(bucket, "data/data/application_test.csv",
                                 "/tmp/application_test.csv")

# ─── Preprocessing (IV‑bin both numeric & categorical) ───────────────────
def preprocess(df_tr, df_te, n_features="auto"):
    num_cols = df_tr.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df_tr.select_dtypes(include=["object","category"]).columns.tolist()
    for drop in ("SK_ID_CURR","TARGET"):
        if drop in num_cols: num_cols.remove(drop)

    def compute_iv(bins, y):
        df = pd.DataFrame({"b": bins, "t": y})
        tot_g = (df.t == 0).sum(); tot_b = (df.t == 1).sum()
        if tot_g == 0 or tot_b == 0: return 0.0
        iv = 0.0
        for _, g in df.groupby("b"):
            good = (g.t==0).sum() or 0.5
            bad  = (g.t==1).sum() or 0.5
            iv += (good/tot_g - bad/tot_b) * np.log((good/tot_g)/(bad/tot_b))
        return iv

    y_tr      = df_tr["TARGET"]
    survivors = []
    for feat in num_cols + cat_cols:
        bp1 = BinningProcess(
            [feat],
            categorical_variables=[feat] if feat in cat_cols else []
        )
        bp1.fit(df_tr[[feat]].values, y_tr)
        bins = bp1.transform(df_tr[[feat]].values).flatten()
        iv   = compute_iv(bins, y_tr)
        miss = df_tr[feat].isna().mean()
        if 0.02 <= iv <= 0.5 and miss <= 0.1:
            survivors.append(feat)

    bp = BinningProcess(
        variable_names=survivors,
        categorical_variables=[c for c in survivors if c in cat_cols]
    )
    bp.fit(df_tr[survivors].values, y_tr)
    df_tr_b = pd.DataFrame(
        bp.transform(df_tr[survivors].values),
        columns=survivors
    )
    df_te_b = pd.DataFrame(
        bp.transform(df_te[survivors].values),
        columns=survivors
    )

    k = len(survivors) if n_features=="auto" else int(n_features)
    selector = SelectKBest(f_classif, k=k)
    selector.fit(df_tr_b.fillna(0), y_tr)
    keep = df_tr_b.columns[selector.get_support()]

    return df_tr_b[keep], df_te_b[keep]

ref_df, test_binned = preprocess(raw_train, raw_test)

# ─── Compute helpers ─────────────────────────────────────────────────────
def compute_drift(ref, cur):
    rpt = Report(metrics=[DataDriftPreset()])
    rpt.run(reference_data=ref, current_data=cur)
    return rpt.as_dict()["metrics"][0]["result"]["dataset_drift"]

def compute_gini(y, p):
    auc = roc_auc_score(y, p)
    return 2*auc - 1

# ─── Prometheus callbacks ─────────────────────────────────────────────────
def drift_cb(options):
    score = compute_drift(ref_df, test_binned)
    return [("app_data_drift_score", score)]

def gini_cb(options):
    if "TARGET" not in raw_test.columns:
        return []
    from app.app import xgb_model, FEATURES
    y     = raw_test["TARGET"].to_numpy()
    p     = xgb_model.predict_proba(test_binned[FEATURES])[:,1]
    return [("app_model_gini", compute_gini(y,p))]

meter.register_observable_instrument(drift_gauge, drift_cb)
meter.register_observable_instrument(gini_gauge,  gini_cb)

# ─── Keep process alive for Prometheus scrapes ───────────────────────────
while True:
    time.sleep(60)
