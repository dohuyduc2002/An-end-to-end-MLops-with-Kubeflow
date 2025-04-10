import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile
import os
import json

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, classification_report
from sklearn.model_selection import train_test_split
import optuna
import mlflow.xgboost
import mlflow.lightgbm
import shap

from src.config import Config, logging  

# =============================================================================
# UnderWritingModel Class (Training & Logging Only)
# =============================================================================
class UnderWritingModel:
    """
    Trains a model using preprocessed/feature-selected data.

    Assumes that the training (and optionally test) datasets are already processed 
    (i.e., produced by your preprocessing/feature selection pipeline). 
    This class uses hyperparameter tuning via Optuna to train either an XGBoost or LightGBM model,
    then logs performance metrics (accuracy, ROC AUC, classification report, SHAP summary plot, 
    and the schema of the training data) to MLflow.
    """
    def __init__(self, X_train_processed, y_train, X_test_processed=None, y_test=None):
        self.X_train_processed = X_train_processed.copy()
        self.y_train = y_train
        self.X_test_processed = X_test_processed.copy() if X_test_processed is not None else None
        self.y_test = y_test
        self.model = None
        self.best_params = None
        self.results = {}

    def train_model(self, model_type="xgb", n_trials=50, random_state=42):
        X_tr, X_val, y_tr, y_val = train_test_split(
            self.X_train_processed, self.y_train, test_size=0.2, random_state=random_state
        )
        if model_type == "xgb":
            def objective_xgb(trial):
                params = {
                    "max_depth": trial.suggest_int("max_depth", 2, 8),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0)
                }
                model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='auc', **params)
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                          early_stopping_rounds=10, verbose=False)
                return accuracy_score(y_val, model.predict(X_val))
            
            study = optuna.create_study(direction="maximize")
            study.optimize(objective_xgb, n_trials=n_trials, show_progress_bar=False)
            self.best_params = study.best_params
            self.model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='auc', **self.best_params)
            self.model.fit(self.X_train_processed, self.y_train)

        elif model_type == "lgbm":
            def objective_lgbm(trial):
                params = {
                    "max_depth": trial.suggest_int("max_depth", 2, 8),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0)
                }
                model = LGBMClassifier(**params)
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                          early_stopping_rounds=10, verbose=-1)
                return accuracy_score(y_val, model.predict(X_val))
            
            study = optuna.create_study(direction="maximize")
            study.optimize(objective_lgbm, n_trials=n_trials, show_progress_bar=False)
            self.best_params = study.best_params
            self.model = LGBMClassifier(**self.best_params)
            self.model.fit(self.X_train_processed, self.y_train)
        else:
            raise ValueError("model_type must be 'xgb' or 'lgbm'.")
        return self.model

    def log_model(self, model_type, version):
        if self.model is None:
            raise RuntimeError("Model is not trained yet.")
        model_name = f"{version}_{'XGBoost' if model_type=='xgb' else 'LightGBM'}"
        
        if self.y_test is not None and self.X_test_processed is not None:
            X_eval = self.X_test_processed
            y_eval = self.y_test
        else:
            X_eval = self.X_train_processed
            y_eval = self.y_train

        preds = self.model.predict(X_eval)
        eval_accuracy = accuracy_score(y_eval, preds)
        eval_report_str = classification_report(y_eval, preds)

        try:
            if hasattr(self.model, "predict_proba"):
                probas = self.model.predict_proba(X_eval)[:, 1]
            else:
                probas = self.model.decision_function(X_eval)
            roc_auc = roc_auc_score(y_eval, probas)
            fpr, tpr, _ = roc_curve(y_eval, probas)
            roc_auc_manual = auc(fpr, tpr)
        except Exception:
            roc_auc, roc_auc_manual = None, None

        logging.info("ROC AUC (roc_auc_score): %s", roc_auc)
        logging.info("ROC AUC (computed with auc & roc_curve): %s", roc_auc_manual)

        # Log classification report using a temporary file (will be deleted after logging)
        with NamedTemporaryFile(mode="w", delete=False, suffix="_classification_report.txt") as tmp_report:
            tmp_report.write(eval_report_str)
            report_path = tmp_report.name

        # Log SHAP summary plot using a temporary file
        import matplotlib.pyplot as plt
        shap_explainer = shap.Explainer(self.model)
        shap_values = shap_explainer(X_eval)
        shap.summary_plot(shap_values, X_eval, show=False)
        fig = plt.gcf()
        with NamedTemporaryFile(suffix="_shap_summary.png", delete=False) as tmp_shap:
            shap_path = tmp_shap.name
            fig.savefig(shap_path)
        plt.close(fig)

        # Save schema mappings of the training data as a JSON file.
        # This creates a dictionary mapping each feature to its numpy dtype as a string.
        schema_mappings = self.X_train_processed.dtypes.apply(lambda x: x.name).to_dict()
        with NamedTemporaryFile(mode="w", delete=False, suffix="_schema_mappings.json") as tmp_schema:
            json.dump(schema_mappings, tmp_schema, indent=4)
            schema_path = tmp_schema.name

        # Log all artifacts and model to MLflow
        with mlflow.start_run(run_name=f"{model_name}_model"):
            mlflow.log_params(self.best_params if self.best_params is not None else self.model.get_params())
            mlflow.log_metric("accuracy", eval_accuracy)
            if roc_auc is not None:
                mlflow.log_metric("roc_auc", roc_auc)
            if roc_auc_manual is not None:
                mlflow.log_metric("roc_auc_manual", roc_auc_manual)
            mlflow.log_artifact(report_path)
            mlflow.log_artifact(shap_path)
            mlflow.log_artifact(schema_path)
            if model_type == "xgb":
                mlflow.xgboost.log_model(self.model, artifact_path="model")
            elif model_type == "lgbm":
                mlflow.lightgbm.log_model(self.model, artifact_path="model")
        # Clean up temporary files
        os.remove(report_path)
        os.remove(shap_path)
        os.remove(schema_path)

        self.results = {
            "accuracy": eval_accuracy,
            "classification_report": eval_report_str
        }

    def explain_model(self, model_type="xgb", data="train"):
        if self.model is None:
            raise RuntimeError("Model is not trained yet.")
        X_explain = self.X_train_processed if data == "train" else self.X_test_processed
        explainer = shap.Explainer(self.model)
        shap_values = explainer(X_explain)
        shap.summary_plot(shap_values, X_explain, show=False)
        return shap_values

# =============================================================================
# UnderWritingTrainer Class
# =============================================================================
class UnderWritingTrainer:
    """
    Wraps the training and logging workflow for the underwriting model.

    This class is designed to work both as a standalone script and in a notebook.
    It requires preprocessed and feature-selected training and test datasets provided as file paths.
    The processed training file must include a TARGET column.
    """
    @staticmethod
    def train_model(model_name="xgb", processed_train=None, processed_test=None, version="v1", experiment_name=None):
        logging.info("MLflow tracking URI: %s", Config.MLFLOW_URI)
        mlflow.set_tracking_uri(Config.MLFLOW_URI)

        if experiment_name:
            mlflow.set_experiment(experiment_name)
            logging.info("MLflow experiment set to: %s", experiment_name)
        
        logging.info("Start training underwriting model %s with version %s", model_name, version)

        df_processed_train = pd.read_csv(processed_train)
        y_train = df_processed_train["TARGET"].values
        X_train_proc = df_processed_train.drop(columns=["TARGET"])
        X_test_proc = pd.read_csv(processed_test) if processed_test is not None else None

        # Optionally, you can also save the schema mappings of the training data here.
        # This is similar to what is logged later in MLflow but can be useful as an independent file.
        schema_mappings = X_train_proc.dtypes.apply(lambda x: x.name).to_dict()
        schema_file = "schema_mappings.json"
        with open(schema_file, "w") as fp:
            json.dump(schema_mappings, fp, indent=4)
        logging.info("Saved schema mappings to %s", schema_file)
        
        trainer = UnderWritingModel(X_train_proc, y_train, X_test_proc, None)

        if model_name.lower() == "xgb":
            trained_model = trainer.train_model(model_type="xgb")
        elif model_name.lower() == "lgbm":
            trained_model = trainer.train_model(model_type="lgbm")
        else:
            raise ValueError("Model name must be 'xgb' or 'lgbm'.")

        trainer.log_model(model_type=model_name.lower(), version=version)
        logging.info("Finished training underwriting model %s with version %s", model_name, version)
        return trained_model

def main():
    parser = argparse.ArgumentParser(description="Train underwriting model")
    parser.add_argument("--model_name", type=str, default="xgb", help="Choose 'xgb' for XGBoost or 'lgbm' for LightGBM")
    parser.add_argument("--processed_train", type=str, required=True,
                        help="Path to preprocessed training CSV file (including 'TARGET' column)")
    parser.add_argument("--processed_test", type=str, required=True,
                        help="Path to preprocessed test CSV file (features only)")
    parser.add_argument("--version", type=str, default="v1", help="Version of the experiment")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Name of the MLflow experiment (if not provided, default is used)")
    subparsers = parser.add_subparsers(dest="sub_model", required=True, help="Model-specific parameters")
    xgb_parser = subparsers.add_parser("xgb", help="XGBoost parameters")
    xgb_parser.add_argument("--xgb_max_depth", type=int, default=6, help="XGB max_depth")
    xgb_parser.add_argument("--xgb_learning_rate", type=float, default=0.1, help="XGB learning rate")
    xgb_parser.add_argument("--xgb_n_estimators", type=int, default=100, help="XGB n_estimators")
    xgb_parser.add_argument("--xgb_subsample", type=float, default=0.8, help="XGB subsample")
    xgb_parser.add_argument("--xgb_colsample_bytree", type=float, default=0.8, help="XGB colsample_bytree")
    lgbm_parser = subparsers.add_parser("lgbm", help="LightGBM parameters")
    lgbm_parser.add_argument("--lgb_max_depth", type=int, default=6, help="LGB max_depth")
    lgbm_parser.add_argument("--lgb_learning_rate", type=float, default=0.1, help="LGB learning rate")
    lgbm_parser.add_argument("--lgb_n_estimators", type=int, default=100, help="LGB n_estimators")
    lgbm_parser.add_argument("--lgb_subsample", type=float, default=0.8, help="LGB subsample")
    lgbm_parser.add_argument("--lgb_colsample_bytree", type=float, default=0.8, help="LGB colsample_bytree")
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    
    model_params = {}
    if args.sub_model == "xgb":
        model_params = {
            "max_depth": args.xgb_max_depth,
            "learning_rate": args.xgb_learning_rate,
            "n_estimators": args.xgb_n_estimators,
            "subsample": args.xgb_subsample,
            "colsample_bytree": args.xgb_colsample_bytree
        }
    elif args.sub_model == "lgbm":
        model_params = {
            "max_depth": args.lgb_max_depth,
            "learning_rate": args.lgb_learning_rate,
            "n_estimators": args.lgb_n_estimators,
            "subsample": args.lgb_subsample,
            "colsample_bytree": args.lgb_colsample_bytree
        }
    logging.info("Model name: %s", args.model_name)
    logging.info("Model-specific parameters: %s", model_params)
    logging.info("Experiment name: %s", args.experiment_name)
    
    trained_model = UnderWritingTrainer.train_model(
        model_name=args.model_name,
        processed_train=args.processed_train,
        processed_test=args.processed_test,
        version=args.version,
        experiment_name=args.experiment_name
    )
    
    # Do not save the final model locally (model file is stored in MLflow)
    # logging.info("Final model is logged in MLflow.")
    
if __name__ == "__main__":
    main()
