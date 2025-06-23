import pandas as pd
import numpy as np
from optbinning import BinningProcess
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import os
import optuna
import xgboost as xgb
from lightgbm import LGBMClassifier

BASE_IMAGE = "microwave1005/kfp_run_image:0.0.1"
TARGET_IMAGE = "docker.io/microwave1005/kfp_run_image:0.0.1"

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

def select_survivors(
    df_train, cat_cols, num_cols, iv_min, iv_max, missing_thres
):
    y = df_train["TARGET"]
    X_train = df_train.drop("TARGET", axis=1)
    survivors = []
    for feature in cat_cols + num_cols:
        missing_rate = X_train[feature].isna().mean()
        if missing_rate > missing_thres:
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
        if iv_min <= iv <= iv_max:
            survivors.append(feature)
    return survivors

def fit_binning(X_train, X_test, y, survivors, cat_cols):
    # After filtering with iv and missing rate, we can proceed with binning
    opt_binning_process = BinningProcess(
        variable_names=survivors,
        categorical_variables=[col for col in survivors if col in cat_cols],
    )
    opt_binning_process.fit(X_train[survivors].values, y)
    df_train_binned = pd.DataFrame(
        opt_binning_process.transform(X_train[survivors].values), columns=survivors
    )
    df_train_binned["TARGET"] = y.values

    # Due to test set does not have TARGET col, we cannot use iv
    df_test_binned = pd.DataFrame(
        opt_binning_process.transform(X_test[survivors].values), columns=survivors
    )
    return opt_binning_process, df_train_binned, df_test_binned

def fit_selector(df_train_binned, df_test_binned, y, n_features_to_select):
    k = (
        len(df_train_binned.columns)
        if n_features_to_select == "auto"
        else int(n_features_to_select)
    )
    selector = SelectKBest(f_classif, k=k)
    selector.fit(df_train_binned.fillna(0), y)
    keep = df_train_binned.columns[selector.get_support()]
    out_train = pd.DataFrame(selector.transform(df_train_binned), columns=keep)
    out_test = pd.DataFrame(selector.transform(df_test_binned), columns=keep)
    out_train["TARGET"] = y
    return selector, out_train, out_test

def build_objective(X, y, model_type, suffix):
    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-3, 0.3, log=True
            ),
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        clf = (
            xgb.XGBClassifier(eval_metric="auc", **params)
            if model_type == "xgb"
            else LGBMClassifier(**params)
        )

        # ----- Hyperparameter optimization -----
        with mlflow.start_run(
            nested=True, run_name=f"optuna_{trial.number}"
        ) as trial_run:
            clf.fit(X_train, y_train)
            acc = accuracy_score(y_val, clf.predict(X_val))

            mlflow.log_params(params)
            mlflow.log_metric("accuracy", acc)

            # log model
            if model_type == "xgb":
                mlflow.xgboost.log_model(clf, artifact_path="model")
            else:
                mlflow.lightgbm.log_model(clf, artifact_path="model")

            # root artifact dir
            art_dir = "/tmp/trial_artifacts"
            os.makedirs(art_dir, exist_ok=True)

            expl = shap.Explainer(clf)
            shap_vals = expl(X_val)
            plt.figure()
            shap.summary_plot(shap_vals, X_val, show=False)
            shap_path = f"{art_dir}/shap.png"
            plt.savefig(shap_path)
            plt.close()

            report_txt = f"{art_dir}/report.txt"
            report = classification_report(y_val, clf.predict(X_val))
            Path(report_txt).write_text(report)

            joblib_path = f"{art_dir}/model_{trial.number}.joblib"
            joblib.dump(clf, joblib_path)

            mlflow.log_artifacts(art_dir, artifact_path="trial_artifacts")

            trial.set_user_attr("mlflow_run_id", trial_run.info.run_id)

        return acc

    return objective

def run_optuna_study(objective, n_trials=5):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    best_trial = study.best_trial
    best_run_id = best_trial.user_attrs["mlflow_run_id"]
    return best_trial, best_run_id
