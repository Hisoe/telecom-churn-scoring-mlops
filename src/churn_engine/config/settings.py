from enum import StrEnum
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    LOCAL = "local"
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="forbid",
    )

    # Runtime Exceution Environment
    ENVIRONMENT: Environment = Field(
        default=Environment.LOCAL,
        description="Target deployment environment",
    )
    PROJECT_NAME: str = Field(
        default="telecom-churn-scoring",
        description="Project workspace identifier",
    )
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging verbosity level",
    )

    # AWS & Storage Infrastructure
    AWS_REGION: str = Field(
        default="ap-southeast-1",
        description="Primary AWS deployment region",
    )
    S3_BUCKET_NAME: str = Field(
        default="telecom-churn-artifacts-local",
        description="S3 bucket name for training data and ONNX model artifacts",
    )
    LOCAL_DATA_DIR: str = Field(
        default="./data",
        description="Local data directory fallback for offline execution",
    )

    # MLflow Tracking Configuration
    MLFLOW_TRACKING_URI: str = Field(
        default="http://127.0.0.1:5000",
        description="MLflow tracking server URI (local container or AWS App Runner/ECS)",
    )
    MLFLOW_EXPERIMENT_NAME: str = Field(
        default="telecom-customer-churn", description="MLflow experiment grouping namespace"
    )
    MODEL_REGISTRY_NAME: str = Field(
        default="telecom-churn_lgbm",
        description="Canonical registered model name in MLflow registry",
    )

    # Inference Serving Configuration (FastAPI / ECS)
    API_HOST: str = Field(default="0.0.0.0", description="FastAPI host binding")
    API_PORT: int = Field(default=8000, ge=1024, le=65535, description="FastAPI port")
    INFERENCE_BATCH_SIZE: int = Field(default=256, ge=1, le=10000)
    CHURN_RISK_THRESHOLD: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Decision boundary threshold for triggering retention actions",
    )


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Instantiate and cache typed settings instance."""
    return AppSettings()
