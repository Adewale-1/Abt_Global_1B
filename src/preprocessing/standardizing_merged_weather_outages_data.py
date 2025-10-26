"""
Standardization Script for Power Outage Prediction Data
========================================================
Applies z-score normalization to continuous features only.
Preserves target variables, metadata, and binary features.
"""

import os
import pandas as pd
import numpy as np


def main():
    # Input from fixed outlier-cleaned dataset
    input_path = "data/ml_ready/merged_weather_outages_2019_2024_outlier_cleaned.csv"
    df = pd.read_csv(input_path, low_memory=False)

    print(f"Loaded {len(df):,} rows × {len(df.columns)} columns\n")

    # Define columns to exclude from standardization
    target_cols = [
        "any_out",
        "num_out_per_day",
        "minutes_out",
        "customers_out",
        "customers_out_mean",
        "cust_minute_area",
        "pct_out_max",
        "pct_out_area",
        "pct_out_area_unified",
        "pct_out_area_covered",
        "pct_out_max_unified",
    ]

    metadata_cols = ["fips_code", "year", "train_mask", "customers_total"]

    # Binary weather features that should not be standardized
    binary_features = [
        "WT01",
        "WT02",
        "WT03",
        "WT04",
        "WT05",
        "WT06",
        "WT08",
        "WT11",
        "heavy_rain",
        "extreme_rain",
        "heat_wave",
        "extreme_heat",
        "freezing",
        "extreme_cold",
        "high_winds",
        "damaging_winds",
        "freeze_thaw_cycle",
        "ice_storm_risk",
        "wet_windy_combo",
        "multiple_extremes",
        "wet_period_indicator",
    ]

    # Get numeric columns
    num_cols = df.select_dtypes(include=np.number).columns

    # Only standardize continuous features
    exclude_all = target_cols + metadata_cols + binary_features
    continuous_cols = [col for col in num_cols if col not in exclude_all]

    print(f"Standardizing {len(continuous_cols)} continuous feature columns")
    print(
        f"Preserving {len([c for c in exclude_all if c in df.columns])} target/metadata/binary columns\n"
    )

    # Standardize only continuous features
    stan_df = df.copy()
    for col in continuous_cols:
        stan_df[col] = (df[col] - df[col].mean()) / df[col].std()

    # Validation: Verify targets and row count unchanged
    assert len(stan_df) == len(df), "ERROR: Row count changed!"
    assert (stan_df["any_out"] == df["any_out"]).all(), "ERROR: Target modified!"

    print("✅ Validation passed:")
    print(f"   Rows: {len(stan_df):,} (unchanged)")
    print(f"   any_out distribution: {stan_df['any_out'].value_counts().to_dict()}")
    print(f"   Outage rate: {stan_df['any_out'].mean():.2%}")
    print(f"   Counties: {stan_df['fips_code'].nunique()}\n")

    output_path = "data/ml_ready/merged_weather_outages_2019_2024_standardized.csv"
    stan_df.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
