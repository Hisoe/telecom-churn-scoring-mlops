import os
from collections.abc import Generator
from pathlib import Path
from typing import Any

import boto3
import mlflow
import pytest
from botocore.client import Config

from churn_engine.config.settings import AppSettings, get_settings


@pytest.fixture(scope="module", autouse=True)
def setup_runtime_env() -> Generator[None, None, None]:
    """Inject runtime environment variables for S3 and MLflow clients."""
    settings = get_settings()

    os.environ["AWS_EC2_METADATA_DISABLED"] = "true"
    os.environ["AWS_DEFAULT_REGION"] = settings.AWS_REGION
    os.environ["AWS_REGION"] = settings.AWS_REGION
    os.environ["AWS_ACCESS_KEY_ID"] = settings.AWS_ACCESS_KEY_ID or "minioadmin"
    os.environ["AWS_SECRET_ACCESS_KEY"] = settings.AWS_SECRET_ACCESS_KEY or "minioadmin"

    endpoint = settings.MLFLOW_S3_ENDPOINT_URL or "http://127.0.0.1:9000"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = endpoint
    os.environ["AWS_ENDPOINT_URL"] = endpoint
    os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(scope="module")
def settings(setup_runtime_env: None) -> AppSettings:
    """Retrieve typed application settings instance."""
    return get_settings()


@pytest.fixture(scope="module")
def s3_client(settings: AppSettings) -> Any:
    """Instantiate authenticated S3 client configured for local MinIO."""
    return boto3.client(
        "s3",
        endpoint_url=settings.AWS_ENDPOINT_URL or "http://127.0.0.1:9000",
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID or "minioadmin",
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY or "minioadmin",
        region_name=settings.AWS_REGION,
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            connect_timeout=5,
            read_timeout=5,
            retries={"max_attempts": 2},
        ),
    )


def test_minio_s3_bucket_connection(settings: AppSettings, s3_client: Any) -> None:
    """Verify MinIO object store is reachable and target bucket exists."""
    response = s3_client.list_buckets()
    bucket_names = [b["Name"] for b in response.get("Buckets", [])]

    assert (
        settings.S3_BUCKET_NAME in bucket_names
    ), f"Bucket '{settings.S3_BUCKET_NAME}' not found in MinIO. Found: {bucket_names}"


def test_mlflow_tracking_and_artifact_logging(
    settings: AppSettings, s3_client: Any, tmp_path: Path
) -> None:
    """Verify MLflow logs parameters/metrics to Postgres and persists artifacts to MinIO."""
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

    dummy_artifact = tmp_path / "model_spec.json"
    dummy_artifact.write_text('{"model_type": "LightGBM", "version": "0.1.0"}', encoding="utf-8")

    with mlflow.start_run(run_name="infra_verification_run") as run:
        mlflow.log_param("telecom_region", "NCR")
        mlflow.log_metric("baseline_roc_auc", 0.8521)
        mlflow.log_artifact(str(dummy_artifact), artifact_path="model_specs")
        run_id = run.info.run_id

    # Verify relational tracking metadata via MLflow Client
    client = mlflow.tracking.MlflowClient()
    run_data = client.get_run(run_id)

    assert run_data.data.params["telecom_region"] == "NCR"
    assert run_data.data.metrics["baseline_roc_auc"] == pytest.approx(0.8521)

    # Verify artifact in S3 bucket
    s3_objects = s3_client.list_objects_v2(
        Bucket=settings.S3_BUCKET_NAME,
        Prefix=f"mlflow/{run_data.info.experiment_id}/{run_id}/artifacts/model_specs",
    )

    keys = [obj["Key"] for obj in s3_objects.get("Contents", [])]
    assert any(
        "model_spec.json" in k for k in keys
    ), f"Artifact 'model_spec.json' not found in MinIO bucket. Found keys: {keys}"
