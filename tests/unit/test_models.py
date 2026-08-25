from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churn_engine.data.generator import TelecomDataGenerator
from churn_engine.features.pipeline import FeatureStorePipeline
from churn_engine.models.trainer import ModelTrainingEngine


@pytest.fixture(scope="module")
def prepared_features() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Generate small feature set for fast training unit tests."""
    generator = TelecomDataGenerator(random_state=42)
    raw_df = generator.generate_batch(num_records=400)

    pipeline = FeatureStorePipeline(temporal_split_ratio=0.80)
    train_raw, test_raw = pipeline.split_out_of_time(raw_df)

    X_train, y_train = pipeline.fit_transform(train_raw)
    X_test, y_test = pipeline.transform(test_raw)

    return X_train, y_train, X_test, y_test


def test_model_training_engine_evaluation_metrics() -> None:
    """Verify evaluation metric calculations produce valid statistical bounds."""
    y_true = np.array([0, 1, 0, 1, 0, 1, 1, 0])
    y_probs = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.85, 0.15])

    metrics = ModelTrainingEngine.evaluate_predictions(y_true, y_probs)

    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert metrics["log_loss"] >= 0.0


def test_lightgbm_training_and_mlflow_logging(
    prepared_features: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
    tmp_path: Path,
) -> None:
    """Ensure LightGBM trains with Optuna and logs artifacts cleanly to SQLite backend."""
    X_train, y_train, X_test, y_test = prepared_features

    # Configure SQLite tracking DB and artifact root inside the pytest tmp_path
    db_path = (tmp_path / "mlflow.db").as_posix()
    test_tracking_uri = f"sqlite:///{db_path}"

    trainer = ModelTrainingEngine(
        tracking_uri=test_tracking_uri,
        experiment_name="test_unit_experiment",
    )

    model, metrics = trainer.train_lightgbm(X_train, y_train, X_test, y_test, n_trials=2)

    assert model is not None
    assert "roc_auc" in metrics
    assert metrics["roc_auc"] > 0.60
    assert len(model.predict(X_test)) == len(y_test)
