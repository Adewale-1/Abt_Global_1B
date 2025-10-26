"""
DEPRECATED: This script is superseded by enhanced_feature_selection.py

This script was run on a broken dataset (5,922 rows with only 3 counties).
The results are INVALID and should not be used for modeling.

Use instead: src/preprocessing/enhanced_feature_selection.py
Results location: results/feature_selection/

Date deprecated: 2025-10-26
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.impute import SimpleImputer

# WARNING: This path points to the corrected dataset, but this script
# does not properly handle class imbalance or exclude target leakage
input_path = "../../data/ml_ready/merged_weather_outages_2019_2024_encoded.csv"
target_col = "any_out"

# Load dataset
print(f"Loading dataset: {input_path}")
df = pd.read_csv(input_path)
df.columns = df.columns.str.strip()  # clean column names
print(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")

# Drop unwanted columns
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])
    print("Dropped 'Unnamed: 0' column")

# Handle missing values in target
if df[target_col].isna().any():
    print(
        f"Found {df[target_col].isna().sum()} missing values in target '{target_col}'"
    )
    df = df.dropna(subset=[target_col])
    print(f"After dropping, dataset has {df.shape[0]} rows")

# Impute missing values in features
feature_cols = df.drop(columns=[target_col]).columns
imputer = SimpleImputer(strategy="mean")
df[feature_cols] = imputer.fit_transform(df[feature_cols])
print(f"Imputed missing values in {len(feature_cols)} features")

# Remove low-variance features
selector = VarianceThreshold(threshold=0.01)
df_features = df.drop(columns=[target_col])
low_variance_cols = df_features.columns[~selector.fit(df_features).get_support()]
df = df.drop(columns=low_variance_cols)
print(f"Removed {len(low_variance_cols)} low-variance columns")


# Remove highly correlated features
def correlation_filter(df, threshold=0.95):
    corr_matrix = df.drop(columns=[target_col]).corr().abs()
    # Use numpy directly
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    df = df.drop(columns=to_drop)
    print(f"Dropping {len(to_drop)} highly correlated columns (threshold={threshold})")
    return df


df = correlation_filter(df, threshold=0.95)


# SelectKBest features
def select_kbest_features(df, target_col, k=50):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
    selector.fit(X, y)
    selected_cols = X.columns[selector.get_support()]
    print(f"Selected {len(selected_cols)} top features using SelectKBest")
    return df[selected_cols.tolist() + [target_col]]


df_kbest = select_kbest_features(df, target_col=target_col, k=50)

print("Final dataset shape:", df_kbest.shape)
print("Columns selected:", df_kbest.columns.tolist())


"""
Selected 46 top features using SelectKBest
Final dataset shape: (5922, 47)
Columns selected: ['AWND', 'PRCP', 'TMAX', 'TMIN', 'WSF2', 'WT01', 'PRCP_lag1d', 'PRCP_lag2d', 'PRCP_lag3d', 'TMAX_lag1d', 'TMAX_lag2d', 'TMAX_lag3d', 'WSF2_lag1d', 'WSF2_lag2d', 'PRCP_3d_sum', 'PRCP_7d_sum', 'PRCP_14d_sum', 'TMAX_14d_max', 'WSF2_3d_max', 'WSF2_7d_max', 'heavy_rain', 'consecutive_rain_days', 'temp_range_daily', 'temp_volatility_3d', 'temp_change_1d', 'heating_degree_days', 'wet_period_indicator', 'moderate_winds', 'wind_acceleration_1d', 'month', 'thermal_stress_index', 'mechanical_stress_index', 'year', 'fips_code', 'num_out_per_day', 'minutes_out', 'customers_out', 'customers_out_mean', 'customers_total', 'train_mask', 'day_of_week', 'season_spring', 'season_summer', 'season_winter', 'county_fips_FIPS:06075', 'county_name_San Diego County', 'any_out']

"""
