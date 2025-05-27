from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Artifact
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import SCIPY_IMAGE


@dsl.component(base_image=SCIPY_IMAGE)
def modeling(
    train_csv: Input[Dataset],
    test_csv: Input[Dataset],
    model_joblib: Output[Artifact],
    registered_model: Output[Artifact],
    mlflow_run_id: Input[Artifact],
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    mlflow_endpoint: str,
    experiment_name: str,
    model_name: str,
    suffix: str,
):
    import os, json, optuna, shap, joblib, matplotlib.pyplot as plt
    import pandas as pd
    from pathlib import Path
    import mlflow, xgboost as xgb
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://{minio_endpoint}"
    os.environ["AWS_ACCESS_KEY_ID"] = minio_access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = minio_secret_key
    os.environ["MLFLOW_ENDPOINT"] = f"http://{mlflow_endpoint}"

    mlflow.set_tracking_uri(f"http://{mlflow_endpoint}")
    mlflow.set_experiment(experiment_name)

    parent_id: str = Path(mlflow_run_id.path).read_text().strip()

    df = pd.read_csv(train_csv.path)
    X, y = df.drop("TARGET", axis=1), df["TARGET"]

    mlflow.end_run()
    with mlflow.start_run(run_id=parent_id):

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
                if model_name == "xgb"
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
                if model_name == "xgb":
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

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=5)

        best_trial = study.best_trial
        best_run_id = best_trial.user_attrs["mlflow_run_id"]

        best_model_uri = f"runs:/{best_run_id}/model"
        registry = mlflow.register_model(best_model_uri, name=f"{model_name}_{suffix}")

    Path(registered_model.path).parent.mkdir(parents=True, exist_ok=True)
    Path(registered_model.path).write_text(
        json.dumps(
            {
                "parent_run": parent_id,
                "best_trial": best_trial.number,
                "best_trial_run": best_run_id,
                "registered": {"name": registry.name, "version": registry.version},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    import kfp.compiler as compiler

    cur = Path(__file__).parent
    dst = cur.parent / "components"
    dst.mkdir(parents=True, exist_ok=True)
    compiler.Compiler().compile(modeling, str(dst / "model.yaml"))
