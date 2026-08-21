import pandas as pd
import pandera as pa
import pytest

from churn_engine.data.generator import TelecomDataGenerator
from churn_engine.schemas.raw_events import CustomerRawRecordSchema


def test_synthetic_data_generator_contract() -> None:
    """Verify synthetic dataset adheres to Pandera schema constraints and statistical boundaries."""
    generator = TelecomDataGenerator(random_state=101)
    df = generator.generate_batch(num_records=1_000)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1_000
    assert df["churn_target"].nunique() == 2
    assert df["churn_target"].mean() > 0.15
    assert df["customer_id"].str.startswith("CUST-").all()
    assert df["total_charges"].isna().sum() == 0


def test_schema_validation_rejection_on_invalid_data() -> None:
    """Ensure Pandera raises SchemaError when invalid bounds or illegal categories are present."""
    generator = TelecomDataGenerator(random_state=101)
    df = generator.generate_batch(num_records=100)

    # Inject negative tenure and illegal contract string
    corrupted_df = df.copy()
    corrupted_df.loc[0, "tenure_months"] = -5
    corrupted_df.loc[1, "contract_type"] = "unlimited-plan"

    with pytest.raises(pa.errors.SchemaErrors):
        CustomerRawRecordSchema.validate(corrupted_df, lazy=True)
