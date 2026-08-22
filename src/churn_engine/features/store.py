import io
from pathlib import Path
from typing import Any

import boto3
import joblib
import pandas as pd
from botocore.client import Config
from loguru import logger

from churn_engine.config.settings import AppSettings, get_settings
from churn_engine.features.pipeline import FeatureStorePipeline


class FeatureStoreService:
    """Persists transformed feature sets and preprocessor artifacts to S3 and local disk."""

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

    def load_latest_raw_data(self) -> pd.DataFrame:
        """Fetch latest ingested Parquet dataset from the S3 raw landing zone."""
        logger.info(f"Listing raw datasets in bucket '{self.settings.S3_BUCKET_NAME}'")
        response = self.s3_client.list_objects_v2(
            Bucket=self.settings.S3_BUCKET_NAME, Prefix="raw/telecom_churn/"
        )

        contents = response.get("Contents", [])
        if not contents:
            raise FileNotFoundError(
                f"No raw Parquet partitions found in bucket '{self.settings.S3_BUCKET_NAME}'."
            )

        # Sort by last modified timestamp to select the newest raw ingestion batch
        latest_obj = sorted(contents, key=lambda x: x["LastModified"], reverse=True)[0]
        s3_key = latest_obj["Key"]
        logger.info(f"Loading raw partition: s3://{self.settings.S3_BUCKET_NAME}/{s3_key}")

        obj_data = self.s3_client.get_object(Bucket=self.settings.S3_BUCKET_NAME, Key=s3_key)
        return pd.read_parquet(io.BytesIO(obj_data["Body"].read()))

    def process_and_persist(self, output_dir: Path | str = "data/features") -> dict[str, str]:
        """Execute end-to-end feature extraction and write to S3 and local disk.

        Args:
            output_dir: Local directory for feature artifact caching.

        Returns:
            Dictionary containing S3 locations of transformed datasets and preprocessor.
        """
        raw_df = self.load_latest_raw_data()
        pipeline = FeatureStorePipeline(temporal_split_ratio=0.80)

        # 1. Strict Out-of-Time split (0.80 Train / 0.20 Test)
        train_raw, test_raw = pipeline.split_out_of_time(raw_df)

        # 2. Fit on train partition, transform test partition
        X_train, y_train = pipeline.fit_transform(train_raw)
        X_test, y_test = pipeline.transform(test_raw)

        # Re-attach target column for downstream estimator consumption
        train_features = X_train.copy()
        train_features["churn_target"] = y_train.to_numpy()

        test_features = X_test.copy()
        test_features["churn_target"] = y_test.to_numpy()

        # 3. Local Disk Caching
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        local_train_path = out_path / "train_features.parquet"
        local_test_path = out_path / "test_features.parquet"
        local_preprocessor_path = out_path / "preprocessor.joblib"

        train_features.to_parquet(local_train_path, index=False)
        test_features.to_parquet(local_test_path, index=False)
        joblib.dump(pipeline.preprocessor, local_preprocessor_path)

        # 4. S3 / MinIO Silver Tier Upload
        s3_locations: dict[str, str] = {}
        artifacts = [
            (
                "train_features",
                local_train_path,
                "features/telecom_churn/train_features.parquet",
            ),
            (
                "test_features",
                local_test_path,
                "features/telecom_churn/test_features.parquet",
            ),
            (
                "preprocessor",
                local_preprocessor_path,
                "artifacts/preprocessors/preprocessor.joblib",
            ),
        ]

        for name, local_file, s3_key in artifacts:
            with open(local_file, "rb") as f:
                self.s3_client.put_object(
                    Bucket=self.settings.S3_BUCKET_NAME,
                    Key=s3_key,
                    Body=f.read(),
                    ContentType="application/octet-stream",
                )
            s3_locations[name] = f"s3://{self.settings.S3_BUCKET_NAME}/{s3_key}"
            logger.success(f"Uploaded {name} -> {s3_locations[name]}")

        return s3_locations
