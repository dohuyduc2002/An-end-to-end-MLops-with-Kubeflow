from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Artifact
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import SCIPY_IMAGE


@dsl.component(base_image=SCIPY_IMAGE)
def preprocess(
    train_csv: Input[Dataset],
    test_csv: Input[Dataset],
    transformer_joblib: Output[Artifact],
    output_train_csv: Output[Dataset],
    output_test_csv: Output[Dataset],
    mlflow_run_id: Output[Artifact],
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    mlflow_endpoint: str,
    parent_run_name: str,
    dest_train_object: str,
    dest_test_object: str,
    n_features_to_select: str,
    data_version: str,
    experiment_name: str,
):

    import os
    import pandas as pd
    import numpy as np
    import joblib
    from pathlib import Path
    import mlflow
    from optbinning import BinningProcess
    from sklearn.feature_selection import SelectKBest, f_classif

    os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://{minio_endpoint}"
    os.environ["AWS_ACCESS_KEY_ID"] = minio_access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = minio_secret_key
    os.environ["MLFLOW_ENDPOINT"] = f"http://{mlflow_endpoint}"
    # Load data

    df_train = pd.read_csv(train_csv.path)
    df_test = pd.read_csv(test_csv.path)

    # Data processing functions
    def get_lists(df):
        numeric = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        category = df.select_dtypes(include=["object"]).columns.tolist()
        for column in ("SK_ID_CURR", "TARGET"):
            if column in numeric:
                numeric.remove(column)
        return category, numeric

    def iv_score(bins, y):
        df = pd.DataFrame({"bins": bins, "target": y})
        total_good, total_bad = (df.target == 0).sum(), (df.target == 1).sum()
        score = 0
        for _, goods in df.groupby("bins"):
            good = (goods.target == 0).sum() or 0.5
            bad = (goods.target == 1).sum() or 0.5
            score += (good / total_good - bad / total_bad) * np.log(
                (good / total_good) / (bad / total_bad)
            )
        return score

    # Feature selection and binning
    cat_cols, num_cols = get_lists(df_train)
    y = df_train["TARGET"]
    X_train, X_test = df_train.drop("TARGET", axis=1), df_test.copy()

    survivors = []
    for feature in cat_cols + num_cols:
        missing_rate = X_train[feature].isna().mean()
        if missing_rate > 0.1:
            continue  # Exclude features with more than 10% missing values

        # Calulate iv scoore by using quantile cut if large bin or
        if feature in cat_cols:
            # pd.factorize for categorical features to label encode them
            bins = pd.factorize(X_train[feature].fillna("missing"))[0]
        else:
            # if numeric feature has more than is high cardinal (more than 10) use qcut, else use cut with quantile is the nunique value
            if X_train[feature].nunique() > 10:
                bins = pd.qcut(
                    X_train[feature].fillna(X_train[feature].median()),
                    10,
                    duplicates="drop",
                    labels=False,
                )
            else:
                bins = pd.cut(
                    X_train[feature].fillna(X_train[feature].median()),
                    bins=X_train[feature].nunique(),
                    labels=False,
                )

        iv = iv_score(bins, y)
        if 0.02 <= iv <= 0.5:
            survivors.append(feature)

    # After filtering with iv and missing rate, we can proceed with binning
    opt_binning_process = BinningProcess(
        variable_names=survivors,
        categorical_variables=[col for col in survivors if col in cat_cols],
    )
    opt_binning_process.fit(X_train[survivors].values, y)
    df_train_binned = pd.DataFrame(
        opt_binning_process.transform(X_train[survivors].values), columns=survivors
    )

    # Due to test set does not have TARGET col, we cannot use iv 
    df_test_binned = pd.DataFrame(
        opt_binning_process.transform(X_test[survivors].values), columns=survivors
    )

    # Feature selection using anova F-test as a score function 
    k = len(survivors) if n_features_to_select == "auto" else int(n_features_to_select)
    selector = SelectKBest(f_classif, k=k)
    selector.fit(df_train_binned.fillna(0), y)

    keep = df_train_binned.columns[selector.get_support()]
    out_train = pd.DataFrame(selector.transform(df_train_binned), columns=keep)
    out_test = pd.DataFrame(selector.transform(df_train_binned), columns=keep)
    out_train["TARGET"] = y

    # Save artifacts to KFP component outputs
    Path(transformer_joblib.path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"opt_binning_process": opt_binning_process, "selector": selector},
        transformer_joblib.path,
    )
    out_train.to_csv(output_train_csv.path, index=False)
    out_test.to_csv(output_test_csv.path, index=False)

    # Prepare artifact names for MLflow
    train_artifact_name = dest_train_object.replace(".csv", f"_{data_version}.csv")
    test_artifact_name = dest_test_object.replace(".csv", f"_{data_version}.csv")
    transformer_artifact_name = f"transformer_{data_version}.joblib"

    # Create temporary directory for MLflow artifacts due to mlflow requirement
    art_dir = Path("/tmp/mlflow_artifacts")
    art_dir.mkdir(parents=True, exist_ok=True)

    # Save artifacts to mlflow
    joblib.dump(
        {"opt_binning_process": opt_binning_process, "selector": selector},
        art_dir / transformer_artifact_name,
    )
    out_train.to_csv(art_dir / train_artifact_name, index=False)
    out_test.to_csv(art_dir / test_artifact_name, index=False)

    # Log to MLflow
    mlflow.set_tracking_uri(os.environ["MLFLOW_ENDPOINT"])
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=parent_run_name) as parent:

        parent_id = parent.info.run_id
        joblib.dump(
            {"opt_binning_process": opt_binning_process, "selector": selector},
            transformer_joblib.path,
        )
        out_train.to_csv(output_train_csv.path, index=False)
        out_test.to_csv(output_test_csv.path, index=False)

        mlflow.log_artifact(transformer_joblib.path, artifact_path="prep")
        mlflow.log_artifact(output_train_csv.path, artifact_path="prep")
        mlflow.log_artifact(output_test_csv.path, artifact_path="prep")

    Path(mlflow_run_id.path).parent.mkdir(parents=True, exist_ok=True)
    Path(mlflow_run_id.path).write_text(parent_id)


if __name__ == "__main__":
    from pathlib import Path
    import kfp.compiler as compiler

    current_dir = Path(__file__).parent
    components_dir = current_dir.parent / "components"
    components_dir.mkdir(parents=True, exist_ok=True)

    compiler.Compiler().compile(
        preprocess,
        str(components_dir / "preprocess.yaml"),
    )
