import io
from typing import Any

import boto3
import pandas as pd
from botocore.client import Config
from loguru import logger

from churn_engine.config.settings import AppSettings, get_settings
from churn_engine.models.trainer import ModelTrainingEngine


class ModelOrchestrationService:
    """Orchestrates multi-model benchmark training and comparative model evaluation."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self.settings = settings or get_settings()
        self.trainer = ModelTrainingEngine(settings=self.settings)
        self.s3_client: Any = boto3.client(
            "s3",
            endpoint_url=self.settings.AWS_ENDPOINT_URL or "http://127.0.0.1:9000",
            aws_access_key_id=self.settings.AWS_ACCESS_KEY_ID or "minioadmin",
            aws_secret_access_key=self.settings.AWS_SECRET_ACCESS_KEY or "minioadmin",
            region_name=self.settings.AWS_REGION,
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
                connect_timeout=5,
                read_timeout=5,
            ),
        )

    def load_feature_partitions(
        self,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Fetch train and test feature tables from S3."""
        logger.info("Loading feature tables from MinIO / S3 Feature Store")

        train_obj = self.s3_client.get_object(
            Bucket=self.settings.S3_BUCKET_NAME,
            Key="features/telecom_churn/train_features.parquet",
        )
        test_obj = self.s3_client.get_object(
            Bucket=self.settings.S3_BUCKET_NAME,
            Key="features/telecom_churn/test_features.parquet",
        )

        train_df = pd.read_parquet(io.BytesIO(train_obj["Body"].read()))
        test_df = pd.read_parquet(io.BytesIO(test_obj["Body"].read()))

        X_train = train_df.drop(columns=["churn_target"])
        y_train = train_df["churn_target"].astype(int)

        X_test = test_df.drop(columns=["churn_target"])
        y_test = test_df["churn_target"].astype(int)

        return X_train, y_train, X_test, y_test

    def run_training_tournament(self, n_trials_per_model: int = 5) -> dict[str, dict[str, float]]:
        """Train LightGBM, XGBoost, and CatBoost; return metrics leaderboard."""
        X_train, y_train, X_test, y_test = self.load_feature_partitions()

        # 1. Execute Bayesian Optimization & Training for all candidate architectures
        _, lgb_metrics = self.trainer.train_lightgbm(
            X_train, y_train, X_test, y_test, n_trials=n_trials_per_model
        )
        _, xgb_metrics = self.trainer.train_xgboost(
            X_train, y_train, X_test, y_test, n_trials=n_trials_per_model
        )
        _, cb_metrics = self.trainer.train_catboost(
            X_train, y_train, X_test, y_test, n_trials=n_trials_per_model
        )

        # 2. Consolidate results into tournament leaderboard
        results = {
            "LightGBM": lgb_metrics,
            "XGBoost": xgb_metrics,
            "CatBoost": cb_metrics,
        }

        leaderboard = pd.DataFrame(results).T.sort_values(by="roc_auc", ascending=False)
        logger.info("\n" + "=" * 80)
        logger.info("MODEL TOURNAMENT LEADERBOARD (Out-of-Time Evaluation):")
        logger.info(f"\n{leaderboard.to_string()}")
        logger.info("=" * 80)

        best_model_name = leaderboard.index[0]
        best_roc_auc = leaderboard.loc[best_model_name, "roc_auc"]
        logger.success(f"Winning Candidate: {best_model_name} with ROC-AUC: {best_roc_auc:.4f}")

        return results
