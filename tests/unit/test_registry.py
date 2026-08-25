from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churn_engine.models.registry import PromotionGateService


class DummyPyFuncModel:
    """Mock PyFunc estimator simulating fast single-row inference."""

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.array([0.42])


def test_benchmark_inference_latency_calculation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify latency benchmarking computes statistical percentiles hermetically."""
    service = PromotionGateService()

    # 1. Mock artifact download to return an ephemeral local path
    dummy_model_dir = tmp_path / "mock_model"
    dummy_model_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "mlflow.artifacts.download_artifacts",
        lambda *args, **kwargs: str(dummy_model_dir),
    )

    # 2. Mock model loader to return deterministic in-memory predictor
    monkeypatch.setattr(
        "mlflow.pyfunc.load_model",
        lambda *args, **kwargs: DummyPyFuncModel(),
    )

    sample_df = pd.DataFrame({"feature_a": [1.0], "feature_b": [2.0]})
    metrics = service.benchmark_inference_latency(
        model_uri="runs:/fake_run_id/model",
        sample_input=sample_df,
        n_iterations=50,
    )

    assert "latency_p50_ms" in metrics
    assert "latency_p95_ms" in metrics
    assert "latency_p99_ms" in metrics
    assert metrics["latency_p99_ms"] >= metrics["latency_p50_ms"]
    assert metrics["latency_p95_ms"] >= metrics["latency_p50_ms"]
