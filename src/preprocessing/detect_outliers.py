import pandas as pd

# Load the dataset
df = pd.read_csv("data/processed/merged_weather_outages_2019_2024_keep_all.csv", low_memory=False)

# Select only numeric columns
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns

# Use IQR to find outliers
Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

outlier_mask = (df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))
outlier_counts = outlier_mask.sum().sort_values(ascending=False)

print("Top 10 columns with most outliers:")
print(outlier_counts.head(10))
