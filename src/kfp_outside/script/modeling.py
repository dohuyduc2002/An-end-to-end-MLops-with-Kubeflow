# scripts/modeling.py
from kfp import dsl
from kfp.dsl import InputPath, Output, Model, Dataset, OutputPath

@dsl.component(base_image="microwave1005/scipy-img:latest",
                packages_to_install=[
                "protobuf==4.25.5",
                "kfp==2.12.1",
                "fastapi==0.104.1",
                "uvicorn[standard]==0.24.0",
                "loguru==0.7.2",
                "joblib==1.3.2",
                "pandas==2.1.3",
                "pytest==7.4.3",
                "numpy==1.24.4",
                "mlflow==2.8.1",
                "matplotlib==3.8.1",
                "pydantic==1.10.8",
                "ortools==9.7.2996",
                "requests==2.31.0",
                "boto3",
                "shap",
                "optuna",
                "optbinning",
                "urllib3",
                "minio",
                "lightgbm",
                "python-dotenv"
            ])
def modeling(
    train_csv: InputPath(Dataset),
    test_csv: InputPath(Dataset),
    model_joblib: Output[Model],
    registered_model: OutputPath(str),
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    model_name: str = "xgb",
    version: str = "v1",
    experiment_name: str = "UnderwritingPipeline",
):
    import os, json, optuna, shap, matplotlib.pyplot as plt, joblib
    import pandas as pd, mlflow, xgboost as xgb
    from lightgbm import LGBMClassifier
    from pathlib import Path
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, classification_report,
        roc_auc_score, roc_curve, auc,
    )
    import mlflow.xgboost, mlflow.lightgbm

    # Configure MLflow → MinIO
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://{minio_endpoint}"
    os.environ["AWS_ACCESS_KEY_ID"]      = minio_access_key
    os.environ["AWS_SECRET_ACCESS_KEY"]  = minio_secret_key

    # Load processed CSV
    df = pd.read_csv(train_csv)
    X, y = df.drop("TARGET", axis=1), df["TARGET"]

    # Optuna tuning
    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
        clf = (
            xgb.XGBClassifier(use_label_encoder=False, eval_metric="auc", **params)
            if model_name == "xgb"
            else LGBMClassifier(**params)
        )
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        clf.fit(X_tr, y_tr)
        return accuracy_score(y_val, clf.predict(X_val))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5)
    best_params = study.best_params

    # Final train
    clf = (
        xgb.XGBClassifier(use_label_encoder=False, eval_metric="auc", **best_params)
        if model_name == "xgb"
        else LGBMClassifier(**best_params)
    )
    clf.fit(X, y)

    # Dump model artifact
    Path(model_joblib.path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_joblib.path)

    # Evaluate & prepare artifacts
    preds  = clf.predict(X)
    acc    = accuracy_score(y, preds)
    report = classification_report(y, preds)
    try:
        proba      = clf.predict_proba(X)[:, 1]
        roc        = roc_auc_score(y, proba)
        fpr, tpr, _ = roc_curve(y, proba)
        roc_manual = auc(fpr, tpr)
    except:
        roc = roc_manual = None

    art_dir = "/tmp/artifacts"
    Path(art_dir).mkdir(parents=True, exist_ok=True)
    (Path(art_dir) / "report.txt").write_text(report)

    explainer = shap.Explainer(clf)
    shap_vals = explainer(X)
    plt.figure()
    shap.summary_plot(shap_vals, X, show=False)
    plt.savefig(f"{art_dir}/shap.png")
    plt.close()

    (Path(art_dir) / "schema.json").write_text(
        json.dumps(X.dtypes.apply(str).to_dict(), indent=2)
    )

    # Log & register via MLflow
    mlflow.set_tracking_uri("http://mlflow.mlflow.svc.cluster.local:5000")
    mlflow.set_experiment(experiment_name)
    run_name = f"{version}_{model_name.upper()}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", acc)
        if roc is not None:
            mlflow.log_metric("roc_auc", roc)
        if roc_manual is not None:
            mlflow.log_metric("roc_auc_manual", roc_manual)

        mlflow.log_artifacts(art_dir, artifact_path="metrics")

        # log model (no need to capture return value)
        if model_name == "xgb":
            mlflow.xgboost.log_model(clf, "model")
        else:
            mlflow.lightgbm.log_model(clf, "model")

        # now register using the artifact URI string
        model_uri = mlflow.get_artifact_uri("model")
        mlflow.register_model(model_uri, run_name)

    # Emit registered model name
    Path(registered_model).write_text(run_name)


if __name__ == "__main__":
    from pathlib import Path
    import kfp.compiler as compiler

    # Define paths using pathlib
    current_dir = Path(__file__).parent
    components_dir = current_dir.parent / "components"
    components_dir.mkdir(parents=True, exist_ok=True)

    # Compile and write the YAML to the components directory
    compiler.Compiler().compile(
        modeling,
        str(components_dir / "model.yaml"),
    )
