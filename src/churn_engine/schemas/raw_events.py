from typing import cast

import pandas as pd
import pandera as pa
from pandera.typing import DateTime, Float, Int, Series, String


class CustomerRawRecordSchema(pa.DataFrameModel):  # type: ignore[misc]
    """Dataframe schema contract for raw customer telemetry and billing events."""

    customer_id: Series[String] = pa.Field(
        str_matches=r"^CUST-\d{8}$",
        nullable=False,
        description="Unique identifier formatted as CUST-XXXXXXXX",
    )
    event_timestamp: Series[DateTime] = pa.Field(
        nullable=False,
        description="Point-in-time timestamp for transaction/event emission",
    )
    tenure_months: Series[Int] = pa.Field(
        ge=0,
        le=120,
        nullable=False,
        description="Customer tenure in active months",
    )
    contract_type: Series[String] = pa.Field(
        isin=["month-to-month", "one-year", "two-year"],
        nullable=False,
        description="Active contract duration type",
    )
    monthly_charges: Series[Float] = pa.Field(
        ge=10.0,
        le=300.0,
        nullable=False,
        description="Current recurring monthly billing charge in USD",
    )
    total_charges: Series[Float] = pa.Field(
        ge=0.0,
        le=36000.0,
        nullable=False,
        description="Cumulative billed amount over entire lifetime",
    )
    support_tickets_last_30d: Series[Int] = pa.Field(
        ge=0,
        le=30,
        nullable=False,
        description="Number of customer support escalations in previous 30 days",
    )
    avg_data_usage_gb: Series[Float] = pa.Field(
        ge=0.0,
        le=2000.0,
        nullable=False,
        description="Average monthly data consumption in Gigabytes",
    )
    roaming_usage_min: Series[Float] = pa.Field(
        ge=0.0,
        le=5000.0,
        nullable=False,
        description="Total roaming voice usage in minutes",
    )
    payment_method: Series[String] = pa.Field(
        isin=[
            "electronic_check",
            "mailed_check",
            "bank_transfer_automatic",
            "credit_card_automatic",
        ],
        nullable=False,
        description="Primary recurring payment channel",
    )
    churn_target: Series[Int] = pa.Field(
        isin=[0, 1],
        nullable=False,
        description="Binary churn outcome label (1 = churned, 0 = active)",
    )

    class Config:
        strict = True
        coerce = True


def validate_raw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate input DataFrame against CustomerRawRecordSchema."""
    validated = CustomerRawRecordSchema.validate(df)
    return cast(pd.DataFrame, validated)
