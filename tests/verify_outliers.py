import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
df_before = pd.read_csv("data/processed/merged_weather_outages_2019_2024_keep_all.csv")
df_after = pd.read_csv("data/ml_ready/merged_weather_outages_2019_2024_outlier_cleaned.csv")

# Select numeric columns
numeric_cols = df_before.select_dtypes(include=["float64", "int64"]).columns

# Compare ranges
changes = []
for col in numeric_cols:
    before_min, before_max = df_before[col].min(), df_before[col].max()
    after_min, after_max = df_after[col].min(), df_after[col].max()
    if (after_max < before_max) or (after_min > before_min):
        changes.append((col, before_min, before_max, after_min, after_max))

print(f"\nColumns affected by capping: {len(changes)}\n")
print("Top 10 columns with range changes:\n")
for col, bmin, bmax, amin, amax in changes[:10]:
    print(f"{col:25} Before Max: {bmax:10.2f} → After Max: {amax:10.2f}")

# Visualize 
cols_to_plot = ["PRCP", "AWND", "weather_severity_score"]

for col in cols_to_plot:
    plt.figure(figsize=(6, 4))
    plt.boxplot(
        [df_before[col].dropna(), df_after[col].dropna()],
        labels=["Before", "After"],
        patch_artist=True
    )
    plt.title(f"{col} - Before vs After Capping")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.show()
