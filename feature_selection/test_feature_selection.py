import pandas as pd
 
url = "https://raw.githubusercontent.com/Adewale-1/Abt_Global_1B/refs/heads/main/data/ml_ready/merged_weather_outages_2019_2024_imputed.csv"

df = pd.read_csv(url)
print("Shape:", df.shape)
print("Columns:", list(df.columns))
print("\nPreview of the data:")
print(df.head())