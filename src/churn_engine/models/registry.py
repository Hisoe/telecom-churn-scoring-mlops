import io
import json
import os
import time
from pathlib import Path
from typing import Any

import boto3
import mlflow
import numpy as np
import pandas as pd
from botocore.client import Config
from loguru import logger
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient

from churn_engine.config.settings import AppSettings, get_settings


class PromotionGateService:
    """Evaluates production suitability and manages Model Registry promotion."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self.settings = settings or get_settings()

        # Set environment credentials for MLflow and boto3 S3 transfer operations
        os.environ["AWS_ACCESS_KEY_ID"] = self.settings.AWS_ACCESS_KEY_ID or "minioadmin"
        os.environ["AWS_SECRET_ACCESS_KEY"] = self.settings.AWS_SECRET_ACCESS_KEY or "minioadmin"
        os.environ["AWS_DEFAULT_REGION"] = self.settings.AWS_REGION
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = (
            self.settings.AWS_ENDPOINT_URL or "http://127.0.0.1:9000"
        )
        os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"

        boto3.setup_default_session(
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            region_name=os.environ["AWS_DEFAULT_REGION"],
        )

        mlflow.set_tracking_uri(self.settings.MLFLOW_TRACKING_URI)
        self.client = MlflowClient(tracking_uri=self.settings.MLFLOW_TRACKING_URI)

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

    def benchmark_inference_latency(
        self, model_uri: str, sample_input: pd.DataFrame, n_iterations: int = 500
    ) -> dict[str, float]:
        """Benchmark single-record inference latency for SLA verification."""
        logger.info(f"Benchmarking inference latency ({n_iterations} samples): {model_uri}")

        # Download artifacts locally first to avoid Windows drive letter scheme parsing bugs
        local_dir = mlflow.artifacts.download_artifacts(
            artifact_uri=model_uri,
            dst_path=None,
            tracking_uri=self.settings.MLFLOW_TRACKING_URI,
        )
        normalized_path = Path(local_dir).as_posix()
        loaded_model = mlflow.pyfunc.load_model(normalized_path)

        # Warm-up pass to trigger JIT / C-extension memory allocation
        single_row = sample_input.iloc[[0]]
        for _ in range(25):
            _ = loaded_model.predict(single_row)

        latencies_ms: list[float] = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            _ = loaded_model.predict(single_row)
            duration_ms = (time.perf_counter() - start) * 1000.0
            latencies_ms.append(duration_ms)

        lat_arr = np.array(latencies_ms)
        p50 = float(np.percentile(lat_arr, 50))
        p95 = float(np.percentile(lat_arr, 95))
        p99 = float(np.percentile(lat_arr, 99))

        logger.info(
            f"Inference Latency Profile -> P50: {p50:.2f}ms | "
            f"P95: {p95:.2f}ms | P99: {p99:.2f}ms"
        )
        return {"latency_p50_ms": p50, "latency_p95_ms": p95, "latency_p99_ms": p99}

    def evaluate_and_promote_champion(
        self,
        model_name: str = "telecom-churn-classifier",
        min_roc_auc: float = 0.70,
        max_p99_latency_ms: float = 20.0,
    ) -> ModelVersion:
        """Find best completed tournament run, validate quality gates, and assign alias."""
        experiment = self.client.get_experiment_by_name(self.settings.MLFLOW_EXPERIMENT_NAME)
        if not experiment:
            raise ValueError(f"Experiment '{self.settings.MLFLOW_EXPERIMENT_NAME}' not found.")

        # Search exclusively for completed runs
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            order_by=["metrics.oot_roc_auc DESC"],
            max_results=10,
        )

        if not runs:
            raise RuntimeError("No completed (FINISHED) runs found in active experiment.")

        best_run = runs[0]
        run_id = best_run.info.run_id
        best_roc_auc = best_run.data.metrics.get("oot_roc_auc", 0.0)
        algorithm = best_run.data.tags.get("algorithm", "Unknown")

        logger.info(
            f"Top Candidate Run: {run_id} | "
            f"Algorithm: {algorithm} | "
            f"OOT ROC-AUC: {best_roc_auc:.4f}"
        )

        # Gate 1: Statistical Performance Check
        if best_roc_auc < min_roc_auc:
            raise ValueError(
                f"Promotion Rejected: OOT ROC-AUC ({best_roc_auc:.4f}) < Threshold ({min_roc_auc})"
            )

        # Gate 2: Operational Latency SLA Check
        model_uri = f"runs:/{run_id}/model"

        test_obj = self.s3_client.get_object(
            Bucket=self.settings.S3_BUCKET_NAME,
            Key="features/telecom_churn/test_features.parquet",
        )
        test_df = pd.read_parquet(io.BytesIO(test_obj["Body"].read()))
        sample_input = test_df.drop(columns=["churn_target"])

        latency_metrics = self.benchmark_inference_latency(
            model_uri=model_uri, sample_input=sample_input
        )

        if latency_metrics["latency_p99_ms"] > max_p99_latency_ms:
            raise ValueError(
                f"Promotion Rejected: P99 Latency ({latency_metrics['latency_p99_ms']:.2f}ms) "
                f"> Max Allowed SLA ({max_p99_latency_ms:.2f}ms)"
            )

        # 3. Register Model Version in MLflow Model Registry
        logger.info(f"Registering model version under '{model_name}'")
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
            tags={
                "algorithm": algorithm,
                "validated_roc_auc": str(best_roc_auc),
                "latency_p99_ms": str(latency_metrics["latency_p99_ms"]),
                "environment": self.settings.ENVIRONMENT,
            },
        )

        # 4. Assign @champion Alias
        self.client.set_registered_model_alias(
            name=model_name,
            alias="champion",
            version=registered_model.version,
        )

        logger.success(
            f"Model Version {registered_model.version} ({algorithm}) promoted to @champion"
        )

        # 5. Persist Deployment Manifest to S3
        deployment_manifest = {
            "model_name": model_name,
            "version": str(registered_model.version),
            "alias": "champion",
            "run_id": run_id,
            "algorithm": algorithm,
            "oot_roc_auc": best_roc_auc,
            "latency_sla": latency_metrics,
            "artifact_uri": registered_model.source,
            "promoted_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        manifest_key = f"artifacts/models/{model_name}/champion_manifest.json"
        self.s3_client.put_object(
            Bucket=self.settings.S3_BUCKET_NAME,
            Key=manifest_key,
            Body=json.dumps(deployment_manifest, indent=2),
            ContentType="application/json",
        )
        logger.success(
            f"Saved deployment manifest: s3://{self.settings.S3_BUCKET_NAME}/{manifest_key}"
        )

        return registered_model
