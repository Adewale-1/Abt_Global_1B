"""
Merge weather ML features with daily outage labels on (fips_code, day),
keeping ALL columns from both inputs and dropping only duplicate-named
columns (except for the join keys 'fips_code' and 'day').

- Normalizes FIPS to 5-digit strings.
- Normalizes day to 'YYYY-MM-DD' (date-only).
- Enforces uniqueness on (fips_code, day).
- Drops overlapping column names from the OUTAGE dataframe (other than the keys),
  so the merged result has a single copy of each name.
- Left-joins outages -> weather so "normal" days are retained.
- Fills core label columns with sensible zeros; everything else is preserved.
"""

from pathlib import Path
import pandas as pd

# === YOUR FILE PATHS ===
WX_FILE = Path(r"C:\Users\Owner\.cursor\ABT Global\Abt_Global_1B\src\data_collection\data\ml_ready\power_outage_ml_features.csv")
OUT_FILE = Path(r"C:\Users\Owner\.cursor\ABT Global\Abt_Global_1B\src\data_collection\data\processed\outages_daily_2019_2024_labels.csv")
OUT_MERGED = Path(r"C:\Users\Owner\.cursor\ABT Global\Abt_Global_1B\src\data_collection\data\processed\merged_weather_outages_2019_2024_keep_all.csv")

# Core outage label columns we typically want filled (others are left as-is)
CORE_LABELS = [
    "any_out",
    "num_out_per_day",
    "minutes_out",
    "customers_out",
    "cust_minute_area",
    "pct_out_max",
    "pct_out_area",
]

def load_weather(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    # Normalize FIPS -> fips_code (5-digit)
    fips_col = "county_fips" if "county_fips" in df.columns else \
               [c for c in df.columns if "fips" in c.lower()][0]
    df["fips_code"] = df[fips_col].astype(str).str[-5:].str.zfill(5)

    # Normalize 'day' to date-only string
    if "day" not in df.columns:
        raise ValueError("Expected a 'day' column in weather dataset.")
    df["day"] = pd.to_datetime(df["day"]).dt.strftime("%Y-%m-%d")

    # Enforce uniqueness on merge keys
    before = len(df)
    df = df.drop_duplicates(subset=["fips_code", "day"])
    after = len(df)
    if after != before:
        print(f"[weather] dropped {before - after} duplicate key rows")

    return df

def load_outages(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"fips_code": "string"}, low_memory=False)

    # Normalize FIPS -> fips_code
    if "fips_code" not in df.columns:
        fips_col = [c for c in df.columns if "fips" in c.lower()][0]
        df["fips_code"] = df[fips_col].astype("string").str[-5:].str.zfill(5)
    else:
        df["fips_code"] = df["fips_code"].astype("string").str.zfill(5)

    # Normalize day; keep original run_start_time_day if present (not a duplicate name)
    if "run_start_time_day" in df.columns:
        day_col = "run_start_time_day"
    elif "day" in df.columns:
        day_col = "day"
    else:
        day_col = [c for c in df.columns if "day" in c.lower()][0]
    df["day"] = pd.to_datetime(df[day_col]).dt.strftime("%Y-%m-%d")

    # Enforce uniqueness on merge keys
    before = len(df)
    df = df.drop_duplicates(subset=["fips_code", "day"])
    after = len(df)
    if after != before:
        print(f"[outages] dropped {before - after} duplicate key rows")

    return df

def merge_keep_all_cols(wx: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
    # Determine overlapping columns (other than join keys) to drop from OUTAGE side
    overlap = (set(wx.columns) & set(out.columns)) - {"fips_code", "day"}
    if overlap:
        print(f"[merge] dropping overlapping columns from outage data to avoid duplicates: {sorted(overlap)}")
        out = out.drop(columns=list(overlap))

    # Now left-join all columns from the (reduced) outage frame
    merged = wx.merge(out, on=["fips_code", "day"], how="left")

    # Fill the core label columns only (if present); everything else is preserved as-is
    int_like = ["any_out", "num_out_per_day", "minutes_out", "customers_out", "cust_minute_area"]
    float_like = ["pct_out_max", "pct_out_area"]

    for c in int_like:
        if c in merged.columns:
            merged[c] = merged[c].fillna(0).astype(int)
    for c in float_like:
        if c in merged.columns:
            merged[c] = merged[c].fillna(0.0)

    return merged

def main():
    wx = load_weather(WX_FILE)
    out = load_outages(OUT_FILE)

    print(f"[weather] rows={len(wx):,}, cols={len(wx.columns)}")
    print(f"[outages] rows={len(out):,}, cols={len(out.columns)}")

    merged = merge_keep_all_cols(wx, out)

    OUT_MERGED.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_MERGED, index=False)

    # Quick QA
    print(f"[merged] rows={len(merged):,}, cols={len(merged.columns)}")
    print(f"[merged] date range: {pd.to_datetime(merged['day']).min()} → {pd.to_datetime(merged['day']).max()}")
    dup = merged.duplicated(["fips_code", "day"]).sum()
    print(f"[merged] duplicate (fips_code, day) rows: {dup}")

    # Optional sanity on any_out if present
    if "any_out" in merged.columns:
        print("[merged] any_out counts:", merged["any_out"].value_counts(dropna=False).to_dict())

    print("[saved]", OUT_MERGED)

if __name__ == "__main__":
    main()
