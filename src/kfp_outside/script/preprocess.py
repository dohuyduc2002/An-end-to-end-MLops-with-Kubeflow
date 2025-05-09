from typing import NamedTuple
from kfp import dsl
from kfp.dsl import InputPath, Output, Model, Dataset

@dsl.component(base_image="microwave1005/scipy-img:latest")
def preprocess(
    train_csv: InputPath(Dataset),       
    test_csv:  InputPath(Dataset),   
    transformer_joblib: Output[Model],    
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    bucket_name: str,
    dest_train_object: str,
    dest_test_object: str,
    n_features_to_select: str = "auto",
    data_version: str = "v1",
) -> NamedTuple("Keys", [("train_key", str), ("test_key", str)]):
    import pandas as pd, numpy as np, joblib
    from pathlib import Path
    from minio import Minio
    from optbinning import BinningProcess
    from sklearn.feature_selection import SelectKBest, f_classif

    # Load artifact CSVs
    df_tr = pd.read_csv(train_csv)
    df_te = pd.read_csv(test_csv)

    # 2) IV‑based filter & binning
    def get_lists(df):
        num = df.select_dtypes(include=["int64","float64"]).columns.tolist()
        cat = df.select_dtypes(include=["object"]).columns.tolist()
        for c in ("SK_ID_CURR","TARGET"):
            if c in num: num.remove(c)
        return cat, num

    def iv_score(bins, y):
        tmp = pd.DataFrame({"b": bins, "t": y})
        tot_g, tot_b = (tmp.t==0).sum(), (tmp.t==1).sum()
        s = 0
        for _, g in tmp.groupby("b"):
            good = (g.t==0).sum() or 0.5
            bad  = (g.t==1).sum() or 0.5
            s += (good/tot_g - bad/tot_b)*np.log((good/tot_g)/(bad/tot_b))
        return s

    cat_cols, num_cols = get_lists(df_tr)
    y = df_tr["TARGET"]
    X_tr, X_te = df_tr.drop("TARGET", axis=1), df_te.copy()

    survivors = []
    for f in cat_cols+num_cols:
        bp_tmp = BinningProcess([f], categorical_variables=[f] if f in cat_cols else [])
        bp_tmp.fit(X_tr[[f]].values, y)
        b = bp_tmp.transform(X_tr[[f]].values).flatten()
        if 0.02 <= iv_score(b,y) <= 0.5 and X_tr[f].isna().mean()<=0.1:
            survivors.append(f)

    bp = BinningProcess(variable_names=survivors,
                        categorical_variables=[c for c in survivors if c in cat_cols])
    bp.fit(X_tr[survivors].values, y)

    df_tr_b = pd.DataFrame(bp.transform(X_tr[survivors].values), columns=survivors)
    df_te_b = pd.DataFrame(bp.transform(X_te[survivors].values), columns=survivors)

    # 3) SelectKBest
    k = len(survivors) if n_features_to_select=="auto" else int(n_features_to_select)
    sel = SelectKBest(f_classif, k=k)
    sel.fit(df_tr_b.fillna(0), y)

    keep = df_tr_b.columns[sel.get_support()]
    out_tr = pd.DataFrame(sel.transform(df_tr_b), columns=keep)
    out_te = pd.DataFrame(sel.transform(df_te_b), columns=keep)
    out_tr["TARGET"] = y

    # Dump transformer
    Path(transformer_joblib.path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"binning_process": bp, "selector": sel}, transformer_joblib.path)

    # Push processed CSVs back to MinIO
    client = Minio(
        minio_endpoint,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        secure=False,
    )
    tr_key = dest_train_object.replace(".csv", f"_{data_version}.csv")
    te_key = dest_test_object.replace(".csv", f"_{data_version}.csv")
    tmp_tr = f"/tmp/{Path(tr_key).name}"
    tmp_te = f"/tmp/{Path(te_key).name}"
    out_tr.to_csv(tmp_tr, index=False)
    out_te.to_csv(tmp_te, index=False)
    client.fput_object(bucket_name, tr_key, tmp_tr)
    client.fput_object(bucket_name, te_key, tmp_te)

    return (tr_key, te_key)

if __name__ == "__main__":
    from pathlib import Path
    import kfp.compiler as compiler

    # Define paths using pathlib
    current_dir = Path(__file__).parent
    components_dir = current_dir.parent / "components"
    components_dir.mkdir(parents=True, exist_ok=True)

    # Compile and write the YAML to the components directory
    compiler.Compiler().compile(
        preprocess,
        str(components_dir / "preprocess.yaml"),
    )