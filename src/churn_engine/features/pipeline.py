from typing import Self

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


class TelecomFeatureEngineer(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Transformer computing non-linear behavioral velocity and friction ratios."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit stub adhering to Scikit-Learn transformer protocol."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Derive behavioral velocity and friction indicators.

        Args:
            X: Input DataFrame containing raw customer records.

        Returns:
            DataFrame augmented with engineered domain features.
        """
        df = X.copy()

        # Prevent zero-division across tenure and usage denominators
        tenure_safe = np.maximum(df["tenure_months"].to_numpy(dtype=float), 0.0) + 1.0
        monthly_charges_safe = np.maximum(df["monthly_charges"].to_numpy(dtype=float), 1.0)
        data_usage_safe = np.maximum(df["avg_data_usage_gb"].to_numpy(dtype=float), 0.0) + 1.0

        # Domain feature ratios
        df["charge_to_tenure_ratio"] = np.round(df["monthly_charges"] / tenure_safe, 4)
        df["ticket_to_tenure_ratio"] = np.round(df["support_tickets_last_30d"] / tenure_safe, 4)
        df["data_per_dollar_ratio"] = np.round(df["avg_data_usage_gb"] / monthly_charges_safe, 4)
        df["roaming_to_data_ratio"] = np.round(df["roaming_usage_min"] / data_usage_safe, 4)

        return df


def build_preprocessor_pipeline() -> ColumnTransformer:
    """Construct an end-to-end ColumnTransformer preprocessor.

    Returns:
        Configured Scikit-Learn ColumnTransformer instance.
    """
    numeric_features = [
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "support_tickets_last_30d",
        "avg_data_usage_gb",
        "roaming_usage_min",
        "charge_to_tenure_ratio",
        "ticket_to_tenure_ratio",
        "data_per_dollar_ratio",
        "roaming_to_data_ratio",
    ]

    categorical_features = [
        "contract_type",
        "payment_method",
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler(with_centering=True, with_scaling=True)),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    drop=None,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


class FeatureStorePipeline:
    """Orchestrates temporal dataset splitting, transformation, and feature extraction."""

    def __init__(self, temporal_split_ratio: float = 0.80) -> None:
        self.temporal_split_ratio = temporal_split_ratio
        self.feature_engineer = TelecomFeatureEngineer()
        self.preprocessor = build_preprocessor_pipeline()
        self.fitted_ = False

    def split_out_of_time(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Perform a strict temporal Out-of-Time (OOT) train/test split.

        Args:
            df: Raw DataFrame containing 'event_timestamp'.

        Returns:
            Tuple of (train_df, test_df).
        """
        sorted_df = df.sort_values(by="event_timestamp").reset_index(drop=True)
        split_idx = int(len(sorted_df) * self.temporal_split_ratio)
        cutoff_time = sorted_df.iloc[split_idx]["event_timestamp"]

        logger.info(
            f"Applying Out-of-Time Cutoff: {cutoff_time} "
            f"(Train Size: {split_idx}, Test Size: {len(sorted_df) - split_idx})"
        )

        train_df = sorted_df.iloc[:split_idx].copy()
        test_df = sorted_df.iloc[split_idx:].copy()

        return train_df, test_df

    def fit_transform(self, train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Engineer domain features and fit preprocessor on training partition.

        Returns:
            Tuple of (X_train_transformed_df, y_train).
        """
        logger.info("Fitting feature engineering pipeline on training partition")
        y_train = train_df["churn_target"].astype(int)

        # 1. Compute interaction terms
        train_augmented = self.feature_engineer.transform(train_df)

        # 2. Fit ColumnTransformer
        X_train_arr = self.preprocessor.fit_transform(train_augmented)
        feature_names = self.preprocessor.get_feature_names_out()

        X_train_df = pd.DataFrame(X_train_arr, columns=feature_names, index=train_df.index)
        self.fitted_ = True
        return X_train_df, y_train

    def transform(self, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Transform test partition using strictly frozen training moments.

        Returns:
            Tuple of (X_test_transformed_df, y_test).
        """
        if not self.fitted_:
            raise RuntimeError("Pipeline must be fitted prior to transform.")

        logger.info("Transforming test partition using fitted preprocessor")
        y_test = test_df["churn_target"].astype(int)

        test_augmented = self.feature_engineer.transform(test_df)
        X_test_arr = self.preprocessor.transform(test_augmented)
        feature_names = self.preprocessor.get_feature_names_out()

        X_test_df = pd.DataFrame(X_test_arr, columns=feature_names, index=test_df.index)
        return X_test_df, y_test
