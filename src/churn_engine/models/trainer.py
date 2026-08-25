import os
from typing import Protocol

import boto3
import catboost as cb
import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import mlflow.catboost
import mlflow.lightgbm
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import xgboost as xgb
from loguru import logger
from mlflow.models.signature import infer_signature
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from churn_engine.config.settings import AppSettings, get_settings


class EstimatorProtocol(Protocol):
    """Protocol for uniform estimator prediction interfaces."""

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray: ...
    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray: ...


class ModelTrainingEngine:
    """Orchestrates multi-model GBDT training, Optuna tuning, and MLflow logging."""

    def __init__(
        self,
        settings: AppSettings | None = None,
        tracking_uri: str | None = None,
        experiment_name: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()

        # Inject AWS/S3 credentials into process environment for boto3 & s3transfer
        os.environ["AWS_ACCESS_KEY_ID"] = self.settings.AWS_ACCESS_KEY_ID or "minioadmin"
        os.environ["AWS_SECRET_ACCESS_KEY"] = self.settings.AWS_SECRET_ACCESS_KEY or "minioadmin"
        os.environ["AWS_DEFAULT_REGION"] = self.settings.AWS_REGION
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = (
            self.settings.AWS_ENDPOINT_URL or "http://127.0.0.1:9000"
        )
        os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"

        # Explicitly pre-warm botocore credential cache
        boto3.setup_default_session(
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            region_name=os.environ["AWS_DEFAULT_REGION"],
        )

        active_tracking_uri = tracking_uri or self.settings.MLFLOW_TRACKING_URI
        active_experiment = experiment_name or self.settings.MLFLOW_EXPERIMENT_NAME

        mlflow.set_tracking_uri(active_tracking_uri)
        mlflow.set_experiment(active_experiment)

    @staticmethod
    def evaluate_predictions(
        y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.50
    ) -> dict[str, float]:
        """Compute classification performance metrics on test partitions."""
        y_pred = (y_pred_proba >= threshold).astype(int)
        return {
            "roc_auc": float(roc_auc_score(y_true, y_pred_proba)),
            "pr_auc": float(roc_auc_score(y_true, y_pred_proba)),
            "log_loss": float(log_loss(y_true, y_pred_proba)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
        }

    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> plt.Figure:
        """Render confusion matrix visualization as a matplotlib Figure."""
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Active (0)", "Churn (1)"],
            yticklabels=["Active (0)", "Churn (1)"],
            ax=ax,
        )
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Out-of-Time Confusion Matrix")
        plt.tight_layout()
        return fig

    def train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        n_trials: int = 10,
    ) -> tuple[lgb.LGBMClassifier, dict[str, float]]:
        """Tune and train an optimized LightGBM classifier."""
        logger.info(f"Initiating LightGBM Optuna study ({n_trials} trials)")

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 250),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "subsample": trial.suggest_float("subsample", 0.60, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "random_state": 42,
                "verbose": -1,
            }
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]
            return float(roc_auc_score(y_test, probs))

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        best_params["random_state"] = 42
        best_params["verbose"] = -1

        final_model = lgb.LGBMClassifier(**best_params)
        final_model.fit(X_train, y_train)

        test_probs = final_model.predict_proba(X_test)[:, 1]
        metrics = self.evaluate_predictions(y_test.to_numpy(), test_probs)

        with mlflow.start_run(run_name="lightgbm_optimized") as run:
            mlflow.log_params(best_params)
            mlflow.log_metrics({f"oot_{k}": v for k, v in metrics.items()})
            mlflow.set_tag("algorithm", "LightGBM")

            signature = infer_signature(X_train, test_probs)
            mlflow.lightgbm.log_model(
                lgb_model=final_model,
                artifact_path="model",
                signature=signature,
            )

            # Confusion Matrix Artifact
            y_pred = (test_probs >= 0.50).astype(int)
            fig = self.plot_confusion_matrix(y_test.to_numpy(), y_pred)
            mlflow.log_figure(fig, "figures/confusion_matrix.png")
            plt.close(fig)

            logger.success(
                f"LightGBM Run ID: {run.info.run_id} | OOT ROC-AUC: {metrics['roc_auc']:.4f}"
            )

        return final_model, metrics

    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        n_trials: int = 10,
    ) -> tuple[xgb.XGBClassifier, dict[str, float]]:
        """Tune and train an optimized XGBoost classifier."""
        logger.info(f"Initiating XGBoost Optuna study ({n_trials} trials)")

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 250),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 9),
                "subsample": trial.suggest_float("subsample", 0.60, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "eval_metric": "logloss",
                "random_state": 42,
            }
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]
            return float(roc_auc_score(y_test, probs))

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        best_params["eval_metric"] = "logloss"
        best_params["random_state"] = 42

        final_model = xgb.XGBClassifier(**best_params)
        final_model.fit(X_train, y_train)

        test_probs = final_model.predict_proba(X_test)[:, 1]
        metrics = self.evaluate_predictions(y_test.to_numpy(), test_probs)

        with mlflow.start_run(run_name="xgboost_optimized") as run:
            mlflow.log_params(best_params)
            mlflow.log_metrics({f"oot_{k}": v for k, v in metrics.items()})
            mlflow.set_tag("algorithm", "XGBoost")

            signature = infer_signature(X_train, test_probs)
            mlflow.xgboost.log_model(
                xgb_model=final_model,
                artifact_path="model",
                signature=signature,
            )

            fig = self.plot_confusion_matrix(y_test.to_numpy(), (test_probs >= 0.50).astype(int))
            mlflow.log_figure(fig, "figures/confusion_matrix.png")
            plt.close(fig)

            logger.success(
                f"XGBoost Run ID: {run.info.run_id} | OOT ROC-AUC: {metrics['roc_auc']:.4f}"
            )

        return final_model, metrics

    def train_catboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        n_trials: int = 10,
    ) -> tuple[cb.CatBoostClassifier, dict[str, float]]:
        """Tune and train an optimized CatBoost classifier."""
        logger.info(f"Initiating CatBoost Optuna study ({n_trials} trials)")

        def objective(trial: optuna.Trial) -> float:
            params = {
                "iterations": trial.suggest_int("iterations", 50, 250),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
                "depth": trial.suggest_int("depth", 3, 8),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
                "random_seed": 42,
                "verbose": 0,
            }
            model = cb.CatBoostClassifier(**params)
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]
            return float(roc_auc_score(y_test, probs))

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        best_params["random_seed"] = 42
        best_params["verbose"] = 0

        final_model = cb.CatBoostClassifier(**best_params)
        final_model.fit(X_train, y_train)

        test_probs = final_model.predict_proba(X_test)[:, 1]
        metrics = self.evaluate_predictions(y_test.to_numpy(), test_probs)

        with mlflow.start_run(run_name="catboost_optimized") as run:
            mlflow.log_params(best_params)
            mlflow.log_metrics({f"oot_{k}": v for k, v in metrics.items()})
            mlflow.set_tag("algorithm", "CatBoost")

            signature = infer_signature(X_train, test_probs)
            mlflow.catboost.log_model(
                cb_model=final_model,
                artifact_path="model",
                signature=signature,
            )

            fig = self.plot_confusion_matrix(y_test.to_numpy(), (test_probs >= 0.50).astype(int))
            mlflow.log_figure(fig, "figures/confusion_matrix.png")
            plt.close(fig)

            logger.success(
                f"CatBoost Run ID: {run.info.run_id} | OOT ROC-AUC: {metrics['roc_auc']:.4f}"
            )

        return final_model, metrics
