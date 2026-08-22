import numpy as np
import pandas as pd
import pytest

from churn_engine.data.generator import TelecomDataGenerator
from churn_engine.features.pipeline import (
    FeatureStorePipeline,
    TelecomFeatureEngineer,
)


@pytest.fixture(scope="module")
def sample_raw_data() -> pd.DataFrame:
    """Generate deterministic batch of synthetic records for feature testing."""
    generator = TelecomDataGenerator(random_state=42)
    return generator.generate_batch(num_records=500)


def test_telecom_feature_engineer_interaction_terms(
    sample_raw_data: pd.DataFrame,
) -> None:
    """Verify engineered domain ratios are computed accurately without nulls."""
    engineer = TelecomFeatureEngineer()
    transformed = engineer.transform(sample_raw_data)

    expected_cols = [
        "charge_to_tenure_ratio",
        "ticket_to_tenure_ratio",
        "data_per_dollar_ratio",
        "roaming_to_data_ratio",
    ]

    for col in expected_cols:
        assert col in transformed.columns
        assert not transformed[col].isna().any()
        assert not np.isinf(transformed[col]).any()


def test_feature_store_pipeline_oot_split_and_transformation(
    sample_raw_data: pd.DataFrame,
) -> None:
    """Verify leak-free Out-of-Time splitting and ColumnTransformer application."""
    pipeline = FeatureStorePipeline(temporal_split_ratio=0.80)

    train_raw, test_raw = pipeline.split_out_of_time(sample_raw_data)

    # Validate strict chronological ordering
    assert train_raw["event_timestamp"].max() <= test_raw["event_timestamp"].min()
    assert len(train_raw) == 400
    assert len(test_raw) == 100

    X_train, y_train = pipeline.fit_transform(train_raw)
    X_test, y_test = pipeline.transform(test_raw)

    # Validate output dimensions and schema stability
    assert X_train.shape[0] == 400
    assert X_test.shape[0] == 100
    assert X_train.shape[1] == X_test.shape[1]
    assert "churn_target" not in X_train.columns
    assert len(y_train) == 400
    assert len(y_test) == 100
    assert not X_train.isna().any().any()
    assert not X_test.isna().any().any()


def test_transform_fails_when_unfitted(sample_raw_data: pd.DataFrame) -> None:
    """Ensure transforming before fitting raises a descriptive RuntimeError."""
    pipeline = FeatureStorePipeline(temporal_split_ratio=0.80)
    with pytest.raises(RuntimeError, match="Pipeline must be fitted"):
        pipeline.transform(sample_raw_data)
