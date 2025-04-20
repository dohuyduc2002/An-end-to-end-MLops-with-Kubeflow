from typing import NamedTuple
from kfp import dsl

@dsl.component(base_image="microwave1005/scipy-img:latest")
def preprocess_and_push(
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    bucket_name: str,
    train_object_name: str,
    test_object_name: str,
    dest_train_object: str,
    dest_test_object: str,
    n_features_to_select: str = "auto",
    data_version: str = "v1",
) -> NamedTuple("OutputKeys", [("train_key", str), ("test_key", str)]):
    import os, pandas as pd, numpy as np
    from pathlib import Path
    from minio import Minio
    from optbinning import BinningProcess
    from sklearn.feature_selection import SelectKBest, f_classif

    client = Minio(minio_endpoint, access_key=minio_access_key,
                   secret_key=minio_secret_key, secure=False)

    tmp = "/tmp/data"
    Path(tmp).mkdir(exist_ok=True)
    local_tr = Path(tmp) / Path(train_object_name).name
    local_te = Path(tmp) / Path(test_object_name).name
    client.fget_object(bucket_name, train_object_name, str(local_tr))
    client.fget_object(bucket_name, test_object_name, str(local_te))

    df_tr = pd.read_csv(local_tr)
    df_te = pd.read_csv(local_te)

    def get_feature_lists(df):
        num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        for c in ("SK_ID_CURR", "TARGET"):
            if c in num_cols: num_cols.remove(c)
        return cat_cols, num_cols

    def compute_iv(bins, target):
        df = pd.DataFrame({"b": bins, "t": target})
        tot_g, tot_b = (df.t == 0).sum(), (df.t == 1).sum()
        iv = 0
        for _, g in df.groupby("b"):
            good, bad = (g.t == 0).sum() or 0.5, (g.t == 1).sum() or 0.5
            iv += (good/tot_g - bad/tot_b) * np.log((good/tot_g) / (bad/tot_b))
        return iv

    cat, num = get_feature_lists(df_tr)
    y_tr = df_tr["TARGET"]
    X_tr, X_te = df_tr.drop("TARGET", axis=1), df_te.copy()

    survivors = []
    for feat in cat + num:
        bp1 = BinningProcess([feat], categorical_variables=[feat] if feat in cat else [])
        bp1.fit(X_tr[[feat]].values, y_tr)
        bins = bp1.transform(X_tr[[feat]].values).flatten()
        if 0.02 <= compute_iv(bins, y_tr) <= 0.5 and X_tr[feat].isna().mean() <= 0.1:
            survivors.append(feat)

    bp = BinningProcess(variable_names=survivors,
                        categorical_variables=[c for c in survivors if c in cat])
    bp.fit(X_tr[survivors].values, y_tr)
    df_tr_b = pd.DataFrame(bp.transform(X_tr[survivors].values), columns=survivors)
    df_te_b = pd.DataFrame(bp.transform(X_te[survivors].values), columns=survivors)

    k = len(survivors) if n_features_to_select == "auto" else int(n_features_to_select)
    selector = SelectKBest(f_classif, k=k)
    selector.fit(df_tr_b.fillna(0), y_tr)

    keep = df_tr_b.columns[selector.get_support()]
    out_tr = pd.DataFrame(selector.transform(df_tr_b), columns=keep)
    out_te = pd.DataFrame(selector.transform(df_te_b), columns=keep)
    out_tr["TARGET"] = y_tr

    train_key = dest_train_object.replace(".csv", f"_{data_version}.csv")
    test_key = dest_test_object.replace(".csv", f"_{data_version}.csv")
    out_tr_path, out_te_path = f"/tmp/{Path(train_key).name}", f"/tmp/{Path(test_key).name}"
    out_tr.to_csv(out_tr_path, index=False)
    out_te.to_csv(out_te_path, index=False)
    client.fput_object(bucket_name, train_key, out_tr_path)
    client.fput_object(bucket_name, test_key, out_te_path)

    return (train_key, test_key)


if __name__ == "__main__":
    import kfp.compiler as compiler
    compiler.Compiler().compile(
        preprocess_and_push,
        "../yaml_components/preprocess_and_push.yaml",
    )
