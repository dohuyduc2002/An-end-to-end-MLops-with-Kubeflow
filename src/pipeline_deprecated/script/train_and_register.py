from kfp import dsl

@dsl.component(base_image="microwave1005/scipy-img:latest")
def train_and_register(
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    bucket_name: str,
    processed_train_key: str,
    processed_test_key: str,
    model_name: str = "xgb",
    version: str = "v1",
    experiment_name: str = "UnderwritingPipeline",
):
    import os, json, optuna, shap, matplotlib.pyplot as plt
    import pandas as pd, mlflow, xgboost as xgb
    from lightgbm import LGBMClassifier
    from tempfile import NamedTemporaryFile, mkdtemp
    from minio import Minio
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, classification_report,
        roc_auc_score, roc_curve, auc,
    )
    import mlflow.xgboost, mlflow.lightgbm

    # ---------- make MinIO the backend for MLflow artifacts ----------
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://{minio_endpoint}"
    os.environ["AWS_ACCESS_KEY_ID"]      = minio_access_key
    os.environ["AWS_SECRET_ACCESS_KEY"]  = minio_secret_key

    # ---------- fetch the two CSVs from MinIO ------------------------
    client   = Minio(minio_endpoint,
                     access_key=minio_access_key,
                     secret_key=minio_secret_key,
                     secure=False)
    tmp_dir  = mkdtemp()
    local_tr = os.path.join(tmp_dir, "train.csv")
    local_te = os.path.join(tmp_dir, "test.csv")
    client.fget_object(bucket_name, processed_train_key, local_tr)
    client.fget_object(bucket_name, processed_test_key,  local_te)

    df_train = pd.read_csv(local_tr)
    df_test  = pd.read_csv(local_te)

    y_train  = df_train["TARGET"]
    X_train  = df_train.drop(columns=["TARGET"])
    X_test   = df_test                            

    # ----------------------------------------------------------------
    class UnderWritingModel:
        def __init__(self, X_tr, y_tr, X_te):
            self.X_train, self.y_train = X_tr, y_tr
            self.X_test                = X_te
            self.best_params, self.model = {}, None

        def train(self, model_type: str):
            X_tr, X_val, y_tr, y_val = train_test_split(
                self.X_train, self.y_train, test_size=0.2, random_state=42
            )

            def objective(trial):
                p = {
                    "max_depth": trial.suggest_int("max_depth", 2, 8),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                    "n_estimators": trial.suggest_int("n_estimators", 100, 300),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                }
                model = (
                    xgb.XGBClassifier(use_label_encoder=False, eval_metric="auc", **p)
                    if model_type == "xgb" else
                    LGBMClassifier(**p)
                )
                model.fit(X_tr, y_tr)
                return accuracy_score(y_val, model.predict(X_val))

            study = optuna.create_study(direction="maximize")
            study.optimize(lambda t: objective(t), n_trials=5)

            self.best_params = study.best_params
            self.model = (
                xgb.XGBClassifier(use_label_encoder=False, eval_metric="auc", **self.best_params)
                if model_type == "xgb" else
                LGBMClassifier(**self.best_params)
            )
            self.model.fit(self.X_train, self.y_train)

        def log_and_register(self, model_type: str, version: str):
            # use a *labelled* set for all metrics
            X_eval, y_eval = self.X_train, self.y_train

            preds = self.model.predict(X_eval)
            acc   = accuracy_score(y_eval, preds)
            report = classification_report(y_eval, preds)

            try:
                probas = self.model.predict_proba(X_eval)[:, 1]
                roc    = roc_auc_score(y_eval, probas)
                fpr, tpr, _ = roc_curve(y_eval, probas)
                roc_manual  = auc(fpr, tpr)
            except Exception:
                roc = roc_manual = None

            # ---------- write artefacts locally ----------------------
            os.makedirs("/tmp/artifacts", exist_ok=True)

            rep_path = "/tmp/artifacts/report.txt"
            with open(rep_path, "w") as f:
                f.write(report)

            shap_values = shap.Explainer(self.model)(X_eval)
            shap_path = "/tmp/artifacts/shap.png"
            plt.figure()
            shap.summary_plot(shap_values, X_eval, show=False)
            plt.savefig(shap_path); plt.close()

            schema_path = "/tmp/artifacts/schema.json"
            with open(schema_path, "w") as f:
                json.dump(X_eval.dtypes.apply(str).to_dict(), f, indent=2)

            # ---------- MLflow logging --------------------------------
            mlflow.set_tracking_uri("http://mlflow.mlflow.svc.cluster.local:5000")
            mlflow.set_experiment(experiment_name)

            with mlflow.start_run(run_name=f"{version}_{model_type.upper()}"):
                mlflow.log_params(self.best_params)
                mlflow.log_metric("accuracy", acc)
                if roc is not None:       mlflow.log_metric("roc_auc", roc)
                if roc_manual is not None: mlflow.log_metric("roc_auc_manual", roc_manual)

                mlflow.log_artifacts("/tmp/artifacts", artifact_path="metrics")

                if model_type == "xgb":
                    mlflow.xgboost.log_model(self.model, "model")
                else:
                    mlflow.lightgbm.log_model(self.model, "model")

                mlflow.register_model(mlflow.get_artifact_uri("model"),
                                      f"{version}_{model_type.upper()}")

    # run the whole thing
    uw = UnderWritingModel(X_train, y_train, X_test)
    uw.train(model_name)
    uw.log_and_register(model_name, version)

# standalone YAML build
if __name__ == "__main__":
    import kfp.compiler as compiler
    compiler.Compiler().compile(
        train_and_register,
        "../yaml_components/trainer.yaml",
    )
