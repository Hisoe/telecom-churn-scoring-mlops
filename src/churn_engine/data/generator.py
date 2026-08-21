from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
from loguru import logger

from churn_engine.schemas.raw_events import validate_raw_dataframe


class TelecomDataGenerator:
    """Deterministic synthetic data generator for telecom customer churn datasets."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.rng = np.random.default_rng(seed=self.random_state)

    def generate_batch(
        self,
        num_records: int = 10_000,
        base_timestamp: datetime | None = None,
    ) -> pd.DataFrame:
        """Generate a validated synthetic telecom dataset with realistic churn mechanics.

        Args:
            num_records: Number of unique customer records to synthesize.
            base_timestamp: Reference point for point-in-time timestamp anchoring.

        Returns:
            Pandera-validated pandas DataFrame.
        """
        logger.info(f"Generating {num_records} records (Seed: {self.random_state})")

        if base_timestamp is None:
            base_timestamp = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)

        # 1. Primary Keys and Temporal Offsets
        customer_ids = [f"CUST-{i:08d}" for i in range(1, num_records + 1)]
        time_offsets_sec = self.rng.integers(0, 90 * 86400, size=num_records)
        event_timestamps = [
            base_timestamp + timedelta(seconds=int(offset)) for offset in time_offsets_sec
        ]

        # 2. Tenure & Vectorized Contract Assignment
        tenure_months = self.rng.integers(0, 73, size=num_records)

        rand_vals = self.rng.uniform(0.0, 1.0, size=num_records)
        contract_types = np.empty(num_records, dtype=object)

        short_mask = tenure_months < 12
        contract_types[short_mask & (rand_vals < 0.70)] = "month-to-month"
        contract_types[short_mask & (rand_vals >= 0.70) & (rand_vals < 0.90)] = "one-year"
        contract_types[short_mask & (rand_vals >= 0.90)] = "two-year"

        long_mask = ~short_mask
        contract_types[long_mask & (rand_vals < 0.30)] = "month-to-month"
        contract_types[long_mask & (rand_vals >= 0.30) & (rand_vals < 0.70)] = "one-year"
        contract_types[long_mask & (rand_vals >= 0.70)] = "two-year"

        # 3. Monthly and Lifetime Billing Charges
        base_charge = self.rng.uniform(20.0, 110.0, size=num_records)
        monthly_charges = np.round(base_charge, 2)
        total_charges = np.round(
            monthly_charges * np.maximum(tenure_months, 1)
            + self.rng.normal(0, 5, size=num_records),
            2,
        )
        total_charges = np.maximum(total_charges, 0.0)

        # 4. Telemetry Metrics and Support Tickets
        support_tickets = self.rng.poisson(lam=1.2, size=num_records)
        support_tickets = np.clip(support_tickets, 0, 15)

        avg_data_usage_gb = np.round(self.rng.gamma(shape=5.0, scale=8.0, size=num_records), 2)
        roaming_usage_min = np.round(self.rng.exponential(scale=35.0, size=num_records), 2)

        payment_methods = self.rng.choice(
            [
                "electronic_check",
                "mailed_check",
                "bank_transfer_automatic",
                "credit_card_automatic",
            ],
            p=[0.35, 0.15, 0.25, 0.25],
            size=num_records,
        )

        # 5. Non-linear Churn Propensity (Log-Odds Formulation)
        log_odds = (
            -1.8
            + (monthly_charges / 50.0) * 0.45
            - (tenure_months / 12.0) * 0.35
            + (support_tickets * 0.40)
            + np.where(contract_types == "month-to-month", 0.65, -0.45)
            + np.where(payment_methods == "electronic_check", 0.30, -0.15)
            + self.rng.normal(0, 0.25, size=num_records)
        )

        churn_probabilities = 1.0 / (1.0 + np.exp(-log_odds))
        churn_targets = (self.rng.uniform(0, 1, size=num_records) < churn_probabilities).astype(int)

        raw_df = pd.DataFrame(
            {
                "customer_id": customer_ids,
                "event_timestamp": pd.to_datetime(event_timestamps),
                "tenure_months": tenure_months.astype(int),
                "contract_type": contract_types,
                "monthly_charges": monthly_charges.astype(float),
                "total_charges": total_charges.astype(float),
                "support_tickets_last_30d": support_tickets.astype(int),
                "avg_data_usage_gb": avg_data_usage_gb.astype(float),
                "roaming_usage_min": roaming_usage_min.astype(float),
                "payment_method": payment_methods,
                "churn_target": churn_targets.astype(int),
            }
        )

        return validate_raw_dataframe(raw_df)
