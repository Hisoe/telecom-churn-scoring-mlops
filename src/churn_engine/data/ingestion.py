import io
from datetime import UTC, datetime
from typing import Any

import boto3
from botocore.client import Config
from loguru import logger

from churn_engine.config.settings import AppSettings, get_settings
from churn_engine.data.generator import TelecomDataGenerator


class DataIngestionService:
    """Service to orchestrate raw data synthesis and partitioned Parquet S3 ingestion."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self.settings = settings or get_settings()
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

    def run_ingestion(
        self,
        num_records: int = 15_000,
        random_state: int = 42,
    ) -> str:
        """Generate, validate, and upload partitioned Parquet data to Object Storage.

        Returns:
            The S3 URI where the ingestion batch is stored.
        """
        generator = TelecomDataGenerator(random_state=random_state)
        df = generator.generate_batch(num_records=num_records)

        partition_date = datetime.now(UTC).strftime("%Y-%m-%d")
        s3_key = f"raw/telecom_churn/dt={partition_date}/records.parquet"

        buffer = io.BytesIO()
        df.to_parquet(buffer, engine="pyarrow", compression="snappy", index=False)
        buffer.seek(0)

        logger.info(
            f"Persisting {len(df)} records ({buffer.getbuffer().nbytes / 1024:.2f} KB) to s3://{self.settings.S3_BUCKET_NAME}/{s3_key}"
        )

        self.s3_client.put_object(
            Bucket=self.settings.S3_BUCKET_NAME,
            Key=s3_key,
            Body=buffer.getvalue(),
            ContentType="application/octet-stream",
        )

        s3_uri = f"s3://{self.settings.S3_BUCKET_NAME}/{s3_key}"
        logger.success(f"Ingestion batch finalized: {s3_uri}")
        return s3_uri
