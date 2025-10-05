# ============================================================
#  Add DST-aware & coverage-adjusted labels to combined outages CSV
#  Input: outages_daily_2019_2024_combined.csv (your combined daily table)
#  Output: same rows + new columns:
#      minutes_in_local_day, snapshot_minutes, snapshots_count,
#      minutes_observed, coverage,
#      pct_out_area_unified (DST-aware),
#      pct_out_area_covered (coverage-adjusted)
#
#  EXAMPLE (PowerShell):
#    python .\make_labels_from_combined.py `
#      --input_csv  "C:\...\outages_daily_2019_2024_combined.csv" `
#      --output_csv "C:\...\outages_daily_2019_2024_with_labels.csv" `
#      --mcc_csv    "C:\...\MCC.csv"
#
#  Notes:
#    - If customers_total is already present in the combined CSV, --mcc_csv is optional.
#    - No calendar expansion, no zero fabrication — this only augments existing rows.
# ============================================================

import argparse
from pathlib import Path
from math import gcd
from functools import reduce
import pandas as pd

# ---------- Timezone map for your 12 counties; others fall back to Eastern ----------
FALLBACK_TZ = "America/New_York"
FIPS_TZ_MAP = {
    "06073": "America/Los_Angeles",
    "06075": "America/Los_Angeles",
    "12086": "America/New_York",
    "12095": "America/New_York",
    "13121": "America/New_York",
    "17031": "America/Chicago",
    "25025": "America/New_York",
    "36061": "America/New_York",
    "48029": "America/Chicago",
    "48201": "America/Chicago",
    "51059": "America/New_York",
    "53033": "America/Los_Angeles",
}

ALLOWED_CADENCES = [5, 10, 15, 30, 60]  # minutes

def _pad_fips(x: pd.Series) -> pd.Series:
    return x.astype("string").str.zfill(5)

def _to_date(x: pd.Series) -> pd.Series:
    return pd.to_datetime(x, errors="coerce").dt.normalize()

def _minutes_in_local_day(fips: str, day_naive: pd.Timestamp) -> int:
    """
    Minutes in the local civil day for that FIPS (handles DST → 23/24/25 hours).
    """
    tz = FIPS_TZ_MAP.get(str(fips), FALLBACK_TZ)
    try:
        start = pd.Timestamp(day_naive).tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")
        end   = (pd.Timestamp(day_naive) + pd.Timedelta(days=1)).tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")
        if pd.isna(start) or pd.isna(end):
            return 1440
        return int((end - start).total_seconds() // 60)
    except Exception:
        return 1440  # safe fallback

def _infer_snapshot_minutes(df: pd.DataFrame) -> int:
    """
    Infer cadence from minutes_out (which is multiples of the cadence).
    Use GCD across positive minutes_out and snap to nearest allowed value.
    """
    mins = pd.to_numeric(df.get("minutes_out"), errors="coerce")
    mins = mins[(mins.notna()) & (mins > 0)].astype(int)
    if mins.empty:
        return 15
    try:
        g = int(reduce(gcd, mins.tolist()))
        allowed = pd.Series(ALLOWED_CADENCES, dtype="int64")
        return int(allowed.iloc[(allowed - g).abs().values.argmin()])
    except Exception:
        return 15

def _load_mcc(mcc_csv: str | None) -> pd.DataFrame | None:
    if not mcc_csv:
        return None
    mcc = pd.read_csv(mcc_csv, dtype={"County_FIPS": "string"})
    mcc = mcc.rename(columns={"County_FIPS": "fips_code", "Customers": "customers_total"})
    mcc["fips_code"] = _pad_fips(mcc["fips_code"])
    mcc["customers_total"] = pd.to_numeric(mcc["customers_total"], errors="coerce")
    return mcc[["fips_code","customers_total"]]

def main():
    ap = argparse.ArgumentParser(description="Create DST-aware and coverage-adjusted outage rate labels.")
    ap.add_argument("--input_csv", required=True, help="Combined outages CSV (e.g., outages_daily_2019_2024_combined.csv)")
    ap.add_argument("--output_csv", required=True, help="Path to write the augmented CSV")
    ap.add_argument("--mcc_csv", default=None, help="Optional MCC.csv for customers_total if not already in the input")
    ap.add_argument("--force_snapshot_minutes", type=int, default=None,
                    help="Override inferred cadence (choose from 5,10,15,30,60)")
    args = ap.parse_args()

    # ---- Load combined CSV ----
    df = pd.read_csv(args.input_csv, dtype={"fips_code":"string"}, parse_dates=["run_start_time_day"])
    df["fips_code"] = _pad_fips(df["fips_code"])
    df["run_start_time_day"] = _to_date(df["run_start_time_day"])

    # Ensure numeric types for the columns we use
    for col in ["minutes_out", "cust_minute_area", "customers_total", "customers_out"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---- Add customers_total if missing and MCC provided ----
    if "customers_total" not in df.columns or df["customers_total"].isna().all():
        mcc = _load_mcc(args.mcc_csv)
        if mcc is not None:
            df = df.merge(mcc, on="fips_code", how="left")

    # ---- DST-aware minutes_in_local_day ----
    df["minutes_in_local_day"] = df.apply(
        lambda r: _minutes_in_local_day(r["fips_code"], r["run_start_time_day"]),
        axis=1
    )

    # ---- snapshot_minutes ----
    if args.force_snapshot_minutes is not None:
        if args.force_snapshot_minutes not in ALLOWED_CADENCES:
            raise ValueError(f"--force_snapshot_minutes must be one of {ALLOWED_CADENCES}")
        snap = int(args.force_snapshot_minutes)
    else:
        snap = _infer_snapshot_minutes(df)
    df["snapshot_minutes"] = snap

    # ---- coverage: snapshots_count, minutes_observed, coverage ----
    # Note: minutes_out represents outage minutes (not all snapshots).
    mo = pd.to_numeric(df.get("minutes_out"), errors="coerce")
    df["snapshots_count"]  = (mo / snap).where(mo.notna(), pd.NA)
    df["minutes_observed"] = (df["snapshots_count"] * snap).where(df["snapshots_count"].notna(), pd.NA)
    # Coverage = observed_minutes / local day length (clip 0..1); will be NaN if minutes_out is NaN
    df["coverage"] = (df["minutes_observed"] / df["minutes_in_local_day"]).clip(lower=0, upper=1)

    # ---- Labels: pct_out_area_unified (DST-aware) & pct_out_area_covered ----
    if "cust_minute_area" in df.columns and "customers_total" in df.columns:
        denom_full = (pd.to_numeric(df["customers_total"], errors="coerce") *
                      pd.to_numeric(df["minutes_in_local_day"], errors="coerce"))
        # Full-day DST-aware rate (primary reporting label)
        valid_full = denom_full.notna() & (denom_full > 0)
        df.loc[valid_full, "pct_out_area_unified"] = df.loc[valid_full, "cust_minute_area"] / denom_full[valid_full]

        # Coverage-adjusted rate (primary training label)
        denom_cov = (pd.to_numeric(df["customers_total"], errors="coerce") *
                     pd.to_numeric(df["minutes_observed"], errors="coerce"))
        valid_cov = denom_cov.notna() & (denom_cov > 0)
        df.loc[valid_cov, "pct_out_area_covered"] = df.loc[valid_cov, "cust_minute_area"] / denom_cov[valid_cov]
    else:
        print("⚠️  Missing 'customers_total' or 'cust_minute_area'; percent labels will remain NaN.")
        df["pct_out_area_unified"] = pd.NA
        df["pct_out_area_covered"] = pd.NA

    # ---- Optional: unified pct_out_max, if customers_total & customers_out exist ----
    if "customers_out" in df.columns and "customers_total" in df.columns:
        denom_max = pd.to_numeric(df["customers_total"], errors="coerce")
        valid_max = denom_max.notna() & (denom_max > 0)
        df.loc[valid_max, "pct_out_max_unified"] = df.loc[valid_max, "customers_out"] / denom_max[valid_max]

    # ---- Suggested training mask (not required for output; handy for quick checks) ----
    if "customers_total" in df.columns:
        df["train_mask"] = (pd.to_numeric(df["customers_total"], errors="coerce") > 0) & (df["coverage"] >= 0.8)

    # ---- Write output ----
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(["fips_code","run_start_time_day"]).to_csv(out, index=False)

    # ---- QA prints ----
    print(f"✓ Wrote {out}  rows={len(df):,}")
    print(f"  snapshot_minutes inferred/forced = {snap}")
    if "pct_out_area_unified" in df.columns:
        print("  pct_out_area_unified non-null:", int(df["pct_out_area_unified"].notna().sum()))
    if "pct_out_area_covered" in df.columns:
        print("  pct_out_area_covered non-null:", int(df["pct_out_area_covered"].notna().sum()))
    if "coverage" in df.columns and df["coverage"].notna().any():
        cov = df["coverage"].dropna()
        print("  coverage stats:",
              {"min": float(cov.min()), "p25": float(cov.quantile(0.25)),
               "median": float(cov.median()), "p75": float(cov.quantile(0.75)),
               "max": float(cov.max())})

if __name__ == "__main__":
    main()
