# ============================================================
#  Combine outages_daily_2019…2024 into ONE CSV (consistent schema)
#  - Strictly matches per-year files: outages_daily_20YYYY.csv
#  - Excludes the output file itself or any *_combined.csv
#  - Harmonizes dtypes (nullable Int64 for integer-like cols)
#  - ISO date strings; zero-padded FIPS
#  - Drops exact duplicates on (fips_code, run_start_time_day)
#  - Optional Parquet; optional 'year' column
# ============================================================

import argparse
import glob
import re
from pathlib import Path
import pandas as pd
from typing import List

# The exact, consistent output schema we want across all years
OUTPUT_COLUMNS = [
    "fips_code",
    "run_start_time_day",
    "any_out",
    "num_out_per_day",
    "minutes_out",
    "customers_out",        # daily peak
    "customers_out_mean",   # daily mean
    "cust_minute_area",     # area under curve (customer-minutes)
    "pct_out_max",          # optional (NaN if not computed for that year)
    "pct_out_area",         # optional (NaN if not computed for that year)
]

# Dtype intents
INT_COLS   = ["any_out", "num_out_per_day", "minutes_out", "customers_out", "cust_minute_area"]
FLOAT_COLS = ["customers_out_mean", "pct_out_max", "pct_out_area"]

YEAR_FILE_RE = re.compile(r"^outages_daily_(20\d{2})\.csv$")  # strict per-year

def read_and_standardize(path: Path, add_year_col: bool = False) -> pd.DataFrame:
    # Read; parse run_start_time_day if present
    header = pd.read_csv(path, nrows=0)
    parse = ["run_start_time_day"] if "run_start_time_day" in header.columns else []
    df = pd.read_csv(path, parse_dates=parse, dtype={"fips_code": "string"})

    # Ensure required columns exist; if missing, create as NA
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    # Keep only the standardized columns
    df = df[OUTPUT_COLUMNS].copy()

    # Types & formatting
    df["fips_code"] = df["fips_code"].astype("string").str.zfill(5)

    # date to ISO yyyy-mm-dd (string)
    df["run_start_time_day"] = pd.to_datetime(df["run_start_time_day"], errors="coerce").dt.strftime("%Y-%m-%d")

    # Cast numerics
    for col in INT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in FLOAT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    # any_out should be 0/1; if entirely NA, infer from other signals
    if df["any_out"].isna().all():
        inferred = (df[["num_out_per_day", "minutes_out", "customers_out", "cust_minute_area"]]
                    .fillna(0).sum(axis=1) > 0).astype("Int64")
        df["any_out"] = inferred

    # Optionally add a year column for debugging
    if add_year_col:
        df["year"] = Path(path).stem  # e.g., outages_daily_2021
        df["year"] = df["year"].str.extract(r"(20\d{2})", expand=False)

    return df

def main():
    ap = argparse.ArgumentParser(description="Combine outages_daily_YYYY.csv into one CSV with consistent schema.")
    ap.add_argument("--input_dir", type=str, required=True,
                    help="Directory containing outages_daily_20YYYY.csv files.")
    ap.add_argument("--output_csv", type=str, default="outages_daily_2019_2024_combined.csv",
                    help="Path to write the combined CSV.")
    ap.add_argument("--write_parquet", action="store_true",
                    help="Also write a Parquet file next to the CSV.")
    ap.add_argument("--add_year_col", action="store_true",
                    help="Include a 'year' column in the combined output (off by default).")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_csv = Path(args.output_csv).resolve()

    # Collect matching files strictly by regex and exclude the output/combined files
    all_candidates = [Path(p) for p in glob.glob(str(input_dir / "outages_daily_*.csv"))]
    paths: List[Path] = []
    for p in sorted(all_candidates):
        name = p.name
        if p.resolve() == out_csv:
            continue
        if "combined" in name.lower():
            continue
        if YEAR_FILE_RE.match(name):  # only outages_daily_20YYYY.csv
            paths.append(p)

    if not paths:
        raise FileNotFoundError(f"No year files found under {input_dir} matching outages_daily_20YYYY.csv")

    print("Combining these files:")
    for p in paths:
        print("  -", p)

    # Read, standardize, stack
    frames = [read_and_standardize(p, add_year_col=args.add_year_col) for p in paths]
    combined = pd.concat(frames, ignore_index=True)

    # Drop exact duplicates at the daily grain; keep first occurrence
    before = len(combined)
    combined = combined.drop_duplicates(subset=["fips_code", "run_start_time_day"], keep="first")
    after = len(combined)

    # Sort
    sort_cols = ["fips_code", "run_start_time_day"]
    if args.add_year_col and "year" in combined.columns:
        sort_cols = ["fips_code", "run_start_time_day", "year"]
    combined = combined.sort_values(sort_cols).reset_index(drop=True)

    # Final column order
    final_cols = OUTPUT_COLUMNS.copy()
    if args.add_year_col and "year" in combined.columns:
        final_cols += ["year"]
    combined = combined[final_cols]

    # Write CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_csv, index=False)

    print(f"✓ Combined {len(paths)} files → {out_csv}")
    print(f"   Rows: {before:,} → {after:,} after de-dup on (fips_code, run_start_time_day)")
    print("   Columns:", final_cols)

    # Optional Parquet
    if args.write_parquet:
        out_parquet = out_csv.with_suffix(".parquet")
        try:
            combined.to_parquet(out_parquet, index=False)
            print(f"✓ Also wrote {out_parquet}")
        except Exception as e:
            print(f"⚠️  Failed to write Parquet: {e}")

if __name__ == "__main__":
    main()
