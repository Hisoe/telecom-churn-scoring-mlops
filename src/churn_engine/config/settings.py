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

    # Runtime Execution Environment
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

    # AWS & Mock S3 Infrastructure
    AWS_REGION: str = Field(
        default="ap-southeast-1",
        description="Primary AWS deployment region",
    )
    AWS_ACCESS_KEY_ID: str | None = Field(
        default=None,
        description="AWS access key (or MinIO root user in local environment)",
    )
    AWS_SECRET_ACCESS_KEY: str | None = Field(
        default=None,
        description="AWS secret key (or MinIO root password in local environment)",
    )
    AWS_ENDPOINT_URL: str | None = Field(
        default=None,
        description="Custom endpoint URL for local MinIO S3 emulation",
    )
    AWS_EC2_METADATA_DISABLED: bool = Field(
        default=False,
        description="Disable IMDS probing when running outside EC2/ECS",
    )
    S3_BUCKET_NAME: str = Field(
        default="telecom-churn-artifacts-local",
        description="Target S3 bucket name for data and model artifacts",
    )
    LOCAL_DATA_DIR: str = Field(
        default="./data",
        description="Local fallback directory for offline dataset caching",
    )

    # PostgreSQL Metadata Store
    POSTGRES_USER: str = Field(default="mlflow")
    POSTGRES_PASSWORD: str = Field(default="mlflowpassword")
    POSTGRES_DB: str = Field(default="mlflow_db")
    POSTGRES_PORT: int = Field(default=5432, ge=1024, le=65535)

    # MLflow Tracking Server Configuration
    MLFLOW_TRACKING_URI: str = Field(
        default="http://127.0.0.1:5000",
        description="MLflow tracking server URI",
    )
    MLFLOW_EXPERIMENT_NAME: str = Field(
        default="telecom-customer-churn",
        description="MLflow experiment grouping namespace",
    )
    MODEL_REGISTRY_NAME: str = Field(
        default="telecom_churn_lgbm",
        description="Canonical registered model name in MLflow registry",
    )
    MLFLOW_S3_ENDPOINT_URL: str | None = Field(
        default=None,
        description="Endpoint override for MLflow S3 artifact backend",
    )
    MLFLOW_S3_IGNORE_TLS: bool = Field(
        default=False,
        description="Disable TLS validation for local HTTP-based MinIO mock",
    )

    # Inference Serving Parameters
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000, ge=1024, le=65535)
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
