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
    df = pd.read_csv("data/processed/merged_weather_outages_2019_2024_keep_all.csv", low_memory=False)
    print(f"Loaded {len(df):,} rows × {len(df.columns)} columns\n")

    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    print(f"Handling outliers in {len(numeric_cols)} numeric columns...\n")

    df_capped = cap_outliers_iqr(df, numeric_cols)

    output_path = "data/ml_ready/merged_weather_outages_2019_2024_outlier_cleaned.csv"
    df_capped.to_csv(output_path, index=False)
    print(f"Outlier-capped dataset saved to:\n   {output_path}")


if __name__ == "__main__":
    main()
