import pandas as pd
import numpy as np


def cap_outliers_iqr(df: pd.DataFrame, columns=None):
    if columns is None:
        columns = df.select_dtypes(include=["float64", "int64"]).columns

    df = df.copy()
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = np.clip(df[col], lower, upper)
    return df


def main():
    input_path = "data/ml_ready/merged_weather_outages_2019_2024_imputed.csv"
    df = pd.read_csv(input_path, low_memory=False)
    print(f"Loaded {len(df):,} rows × {len(df.columns)} columns\n")

    # Exclude target variables and metadata from outlier handling
    exclude_cols = [
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
        "fips_code",
        "year",
        "train_mask",
        "customers_total",
    ]

    # Only cap outliers in feature columns
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]

    print(f"Handling outliers in {len(feature_cols)} feature columns")
    print(f"Excluding {len(exclude_cols)} target/metadata columns from capping...\n")

    df_capped = cap_outliers_iqr(df, feature_cols)

    # Validation: Verify target unchanged
    assert (
        df_capped["any_out"] == df["any_out"]
    ).all(), "ERROR: Target variable was modified!"
    print("✅ Validation: Target variable preserved")
    print(f"   any_out distribution: {df_capped['any_out'].value_counts().to_dict()}")
    print(f"   Outage rate: {df_capped['any_out'].mean():.2%}\n")

    output_path = "data/ml_ready/merged_weather_outages_2019_2024_outlier_cleaned.csv"
    df_capped.to_csv(output_path, index=False)
    print(f"Outlier-capped dataset saved to:\n   {output_path}")


if __name__ == "__main__":
    main()
