# ============================================================
#  EAGLE-I Outages (ANY YEAR) → Daily rollups → CSV (LOCAL)
#  - Filters to 12 specific counties (FIPS list below)
#  - Uses per-county LOCAL day bucketing (DST-aware)
#  - Auto-detects schema differences (e.g., 2023 has `sum`)
#  - Auto-infers snapshot interval minutes (snapped to {5,10,15,30,60})
#  - Streams CSV in chunks with the C engine (pyarrow has no chunksize)
#  - Ensures consistent output columns/order for easy concatenation
#  - Computes outage metrics from POSITIVE snapshots only (sparse raw)
# ============================================================

import re
from pathlib import Path
import pandas as pd

# ----------------------a
# CONFIG — EDIT THESE PATHS IF NEEDED
# ----------------------
OUTAGE_FILE = r"C:\Users\Owner\.cursor\ABT Global\Abt_Global_1B\src\data_collection\data\raw_outage_data\eaglei_outages_2024.csv"
MCC_FILE    = r"C:\Users\Owner\.cursor\ABT Global\Abt_Global_1B\src\data_collection\data\raw_outage_data\MCC.csv"  # or None
OUT_DIR     = r"C:\Users\Owner\.cursor\ABT Global\Abt_Global_1B\src\data_collection\data\processed"

# Filter to these 12 counties (exact FIPS)
TARGET_FIPS = [
    "06073", "06075", "12086", "12095", "13121", "17031",
    "25025", "36061", "48029", "48201", "51059", "53033",
]

# Time bucketing mode
DAY_TZ_MODE = "per_county"                 # "utc" | "single" | "per_county"
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
# If DAY_TZ_MODE == "single", set this:
LOCAL_TZ = "America/New_York"

# Event gap minutes (contiguous nonzero snapshots are one event)
GAP_MINUTES = 30

# CSV reading performance knobs
USE_ARROW_ENGINE = True            # parquet speed; CSV still uses C engine for chunking
CHUNKSIZE  = 2_000_000             # increase if you have RAM

# Snapshot minutes; if None, auto-infer & snap to {5,10,15,30,60}
SNAP_MINUTES = None

# Enforce consistent output schema/order across years for easy concatenation
ENFORCE_COLUMN_ORDER = True
OUTPUT_COLUMNS = [
    "fips_code",
    "run_start_time_day",
    "any_out",
    "num_out_per_day",
    "minutes_out",
    "customers_out",        # daily peak
    "customers_out_mean",   # daily mean
    "cust_minute_area",     # area under curve (customer-minutes)
    "pct_out_max",          # optional (via MCC or raw total_customers)
    "pct_out_area",         # optional (via MCC or raw total_customers)
]

# -----------------------------------------------------------
# Schema detection
# -----------------------------------------------------------
# Required canonical columns
REQ_KEYS = {
    "fips_code": [
        "fips_code", "county_fips", "fips", "fips5", "county_fips_code", "County_FIPS"
    ],
    "customers_out": [
        # 2019–2022, 2024
        "customers_out",
        # 2023 uses 'sum'
        "sum",
        # other possible vendor synonyms
        "customers_affected", "customers_interrupted", "customers_outstanding",
        "outage_customers", "affected_customers", "customer_out", "cust_out", "n_customers_out"
    ],
    "run_start_time": [
        "run_start_time", "run_time", "interval_start_time", "interval_start",
        "timestamp_utc", "start_time_utc", "start_time", "observation_time_utc",
        "time_utc", "timestamp", "datetime_utc"
    ],
}

# Optional canonical columns (don’t fail if missing)
OPT_KEYS = {
    # 2024 raw sometimes includes this; use as MCC fallback for % metrics
    "customers_total": ["total_customers", "customers_total", "Customers"]
}

def _detect_csv_columns(path: Path) -> list[str]:
    try:
        cols = pd.read_csv(path, nrows=0).columns.tolist()
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV header from {path}: {e}")
    return cols

def _detect_parquet_columns(path: Path) -> list[str]:
    try:
        import pyarrow  # noqa: F401
        cols = pd.read_parquet(path, engine="pyarrow").columns.tolist()
    except Exception:
        cols = pd.read_parquet(path).columns.tolist()
    return cols

def _build_column_mapping(all_cols: list[str]) -> dict:
    """
    Map canonical names → actual file columns (case-insensitive).
    - Enforces presence of REQ_KEYS
    - Adds OPT_KEYS if present
    """
    lower_to_actual = {c.lower(): c for c in all_cols}
    mapping = {}
    missing = []

    # required
    for canon, options in REQ_KEYS.items():
        found = None
        for opt in options:
            if opt.lower() in lower_to_actual:
                found = lower_to_actual[opt.lower()]
                break
        if found is None:
            missing.append((canon, options))
        else:
            mapping[canon] = found

    if missing:
        msg = " | ".join([f"{k}: any of {opts}" for k, opts in missing])
        raise ValueError(
            "Could not detect required columns. "
            f"Missing canonical→options: {msg}. "
            f"Available columns: {all_cols}"
        )

    # optional
    for canon, options in OPT_KEYS.items():
        for opt in options:
            if opt.lower() in lower_to_actual:
                mapping[canon] = lower_to_actual[opt.lower()]
                break

    return mapping

# -----------------------------------------------------------
# IO helpers
# -----------------------------------------------------------
def _read_minimal_outages(path, target_fips, mapping, chunksize):
    """
    Read CSV/Parquet using detected column names.
    Returns canonical columns:
      required: fips_code, customers_out, run_start_time
      optional: customers_total (if present in raw; e.g., 2024)
    """
    p = Path(path)
    target = set(target_fips)
    usecols_actual = [mapping["fips_code"], mapping["customers_out"], mapping["run_start_time"]]
    if "customers_total" in mapping:
        usecols_actual.append(mapping["customers_total"])

    def _rename_and_cast(df):
        rename_map = {
            mapping["fips_code"]: "fips_code",
            mapping["customers_out"]: "customers_out",
            mapping["run_start_time"]: "run_start_time",
        }
        if "customers_total" in mapping:
            rename_map[mapping["customers_total"]] = "customers_total"
        df = df.rename(columns=rename_map)

        df["fips_code"] = df["fips_code"].astype("string").str.zfill(5)
        df["customers_out"] = pd.to_numeric(df["customers_out"], errors="coerce").astype("Int64")
        if "customers_total" in df.columns:
            df["customers_total"] = pd.to_numeric(df["customers_total"], errors="coerce").astype("Int64")
        return df

    if p.suffix.lower() == ".parquet":
        try:
            df = pd.read_parquet(p, columns=usecols_actual, engine="pyarrow" if USE_ARROW_ENGINE else None)
        except Exception:
            df = pd.read_parquet(p, columns=usecols_actual)
        df = _rename_and_cast(df)
        return df[df["fips_code"].isin(target)].copy()

    # CSV (chunked, C engine)
    chunks = []
    reader = pd.read_csv(
        p, usecols=usecols_actual, parse_dates=[mapping["run_start_time"]],
        chunksize=chunksize, low_memory=True  # default C engine
    )
    for chunk in reader:
        chunk = _rename_and_cast(chunk)
        chunk = chunk[chunk["fips_code"].isin(target)]
        if not chunk.empty:
            chunks.append(chunk)

    if not chunks:
        cols = ["fips_code", "customers_out", "run_start_time"]
        if "customers_total" in mapping:
            cols.append("customers_total")
        dtypes = {"fips_code": "string", "customers_out": "Int64"}
        if "customers_total" in mapping:
            dtypes["customers_total"] = "Int64"
        return pd.DataFrame(columns=cols).astype(dtypes)

    return pd.concat(chunks, ignore_index=True)

# -----------------------------------------------------------
# Time handling & rollups
# -----------------------------------------------------------
def _compute_ts_local_and_day(df, day_mode, single_tz, fips_tz_map, fallback_tz):
    """Returns ts_local (tz-aware, mixed zones) and run_start_time_day (tz-naive local date) aligned to chosen mode."""
    # Ensure UTC tz-aware
    df["run_start_time"] = pd.to_datetime(df["run_start_time"], errors="coerce", utc=True)

    if day_mode == "utc":
        ts_local = df["run_start_time"].dt.tz_convert("UTC")
        day = ts_local.dt.floor("D").dt.tz_localize(None)
        return ts_local, day

    if day_mode == "single":
        ts_local = df["run_start_time"].dt.tz_convert(single_tz)
        day = ts_local.dt.floor("D").dt.tz_localize(None)
        return ts_local, day

    # per_county
    tz_series = df["fips_code"].map(fips_tz_map).fillna(fallback_tz)
    df_local = df.copy()
    df_local["_tz"] = tz_series.astype("string")

    ts_local = pd.Series(index=df_local.index, dtype="datetime64[ns, UTC]")
    day      = pd.Series(index=df_local.index, dtype="datetime64[ns]")

    for tz in df_local["_tz"].dropna().unique():
        mask = df_local["_tz"] == tz
        loc = df_local.loc[mask, "run_start_time"].dt.tz_convert(tz)
        ts_local.loc[mask] = loc
        day.loc[mask] = loc.dt.floor("D").dt.tz_localize(None)

    return ts_local, day

def infer_snapshot_minutes(df: pd.DataFrame) -> int:
    """
    Infer snapshot cadence and snap to nearest of {5,10,15,30,60} minutes.
    Robust to sparse/event-only feeds where some ticks are missing.
    """
    diffs = (df.sort_values(["fips_code", "run_start_time"])
               .groupby("fips_code")["run_start_time"]
               .diff().dt.total_seconds().dropna() / 60.0)
    if diffs.empty:
        return 15
    est = diffs.median()
    allowed = pd.Series([5, 10, 15, 30, 60], dtype="float64")
    return int(allowed.iloc[(allowed - est).abs().values.argmin()])

def load_mcc(path_or_none):
    if not path_or_none:
        return None
    try:
        mcc = pd.read_csv(path_or_none, dtype={"County_FIPS": "string"},
                          engine="pyarrow" if USE_ARROW_ENGINE else None)
    except Exception:
        mcc = pd.read_csv(path_or_none, dtype={"County_FIPS": "string"})
    mcc = mcc.rename(columns={"County_FIPS": "fips_code", "Customers": "customers_total"})
    mcc["fips_code"] = mcc["fips_code"].astype("string").str.zfill(5)
    return mcc

def write_csv(df, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def _minutes_in_local_day(fips: str, day_naive_ts: pd.Timestamp) -> int:
    """
    Return the number of minutes in the given LOCAL day for that FIPS (handles DST 23/25h days).
    day_naive_ts must be a pandas Timestamp (date at midnight, tz-naive).
    """
    tz = FIPS_TZ_MAP.get(fips, FALLBACK_TZ)
    start = pd.Timestamp(day_naive_ts).tz_localize(tz)
    end   = (pd.Timestamp(day_naive_ts) + pd.Timedelta(days=1)).tz_localize(tz)
    return int((end - start).total_seconds() // 60)

def process_outages(path, target_fips, day_mode, single_tz, fips_tz_map, fallback_tz,
                    gap_minutes, mcc=None, snap_minutes=None, chunksize=CHUNKSIZE):
    """
    Process an EAGLE-I outages file (CSV or Parquet) into daily rollups for target FIPS.
    Returns (daily_df, snapshot_minutes).
    """
    p = Path(path)
    # Detect schema (actual column names in file)
    all_cols = _detect_parquet_columns(p) if p.suffix.lower() == ".parquet" else _detect_csv_columns(p)
    mapping = _build_column_mapping(all_cols)

    df = _read_minimal_outages(path, target_fips, mapping, chunksize)
    if df.empty:
        raise ValueError(f"No rows found for target FIPS in the outage file: {path}")

    # Normalize time → UTC; compute local day for bucketing
    ts_local, day = _compute_ts_local_and_day(df, day_mode, single_tz, fips_tz_map, fallback_tz)
    df["ts_local"] = ts_local
    df["run_start_time_day"] = day
    df["ts_utc"] = df["run_start_time"]  # stable dtype for gap math

    # Auto-infer snapshot minutes if not supplied
    minutes = snap_minutes if snap_minutes is not None else infer_snapshot_minutes(df)

    # -------- Daily rollups (from POSITIVE snapshots only semantics) --------
    # Sparse/event-only feed means rows already represent >0 outages.
    # But stay safe if a future file includes zeros: use >0 guards.
    def any_nonzero(s):    # -> 0/1 per group
        return int((s.fillna(0) > 0).any())

    def positive_count(s): # -> # of positive snapshots
        return int((s.fillna(0) > 0).sum())

    # Customer-minute "area" per snapshot (minutes weighted)
    df["cust_x_interval"] = df["customers_out"] * minutes  # zero rows (if any) contribute 0

    daily = (
        df.groupby(["fips_code", "run_start_time_day"])
          .agg(
              any_out=("customers_out", any_nonzero),
              minutes_out=("customers_out", lambda s: minutes * positive_count(s)),
              customers_out_max=("customers_out", "max"),
              customers_out_mean=("customers_out", "mean"),  # mean over rows present (sparse => outage-only)
              cust_minute_area=("cust_x_interval", "sum"),
          )
          .reset_index()
    )

    # -------- Event counts from POSITIVE snapshots only --------
    df_pos = df[df["customers_out"].fillna(0) > 0].copy()
    if not df_pos.empty:
        df_pos = df_pos.sort_values(["fips_code", "ts_utc"])
        df_pos["prev_time"] = df_pos.groupby("fips_code")["ts_utc"].shift()
        gap = (df_pos["ts_utc"] - df_pos["prev_time"]).dt.total_seconds().div(60)
        new_event = (
            df_pos["prev_time"].isna()
            | (gap > gap_minutes)
            | (df_pos["run_start_time_day"] != df_pos.groupby("fips_code")["run_start_time_day"].shift())
        )
        df_pos["event_id"] = new_event.astype(int).groupby(df_pos["fips_code"]).cumsum()

        events_daily = (
            df_pos.groupby(["fips_code", "run_start_time_day"])["event_id"]
                  .nunique()
                  .reset_index(name="num_out_per_day")
        )
        daily = daily.merge(events_daily, on=["fips_code", "run_start_time_day"], how="left")
    else:
        daily["num_out_per_day"] = pd.NA

    daily["num_out_per_day"] = daily["num_out_per_day"].fillna(0).astype("Int64")

    # -------- % metrics: prefer MCC; else use raw customers_total if present --------
    denom_col = None
    if mcc is not None and not mcc.empty:
        m = mcc[["fips_code", "customers_total"]]
        daily = daily.merge(m, on="fips_code", how="left")
        denom_col = "customers_total"
    elif "customers_total" in df.columns:
        # derive per-day denominator from raw (use max per day)
        denom = (
            df.groupby(["fips_code", "run_start_time_day"])["customers_total"]
              .max()
              .reset_index(name="customers_total")
        )
        daily = daily.merge(denom, on=["fips_code", "run_start_time_day"], how="left")
        denom_col = "customers_total"

    # Compute DST-aware local minutes-per-day for area denominator
    daily["run_start_time_day"] = pd.to_datetime(daily["run_start_time_day"])  # ensure datetime
    daily["minutes_in_day_local"] = daily.apply(
        lambda r: _minutes_in_local_day(str(r["fips_code"]), r["run_start_time_day"]),
        axis=1
    )

    if denom_col:
        # guard: valid denominators only
        valid = daily[denom_col].fillna(0) > 0
        daily.loc[valid, "pct_out_max"]  = daily.loc[valid, "customers_out_max"] / daily.loc[valid, denom_col]
        daily.loc[valid, "pct_out_area"] = (
            daily.loc[valid, "cust_minute_area"] /
            (daily.loc[valid, denom_col] * daily.loc[valid, "minutes_in_day_local"])
        )

    # Rename to final schema (daily peak as customers_out)
    daily = daily.rename(columns={"customers_out_max": "customers_out"})
    daily["fips_code"] = daily["fips_code"].astype("string").str.zfill(5)

    # ---- Strict file-year filter (avoid spillover around New Year due to local TZ day bucketing)
    fname = Path(path).name
    year_match = re.search(r"(20\d{2})", fname)
    if year_match:
        yr = int(year_match.group(1))
        daily = daily[daily["run_start_time_day"].dt.year == yr]

    # Day to ISO string for CSV output
    daily["run_start_time_day"] = pd.to_datetime(daily["run_start_time_day"]).dt.strftime("%Y-%m-%d")

    # Drop helper column not part of official schema
    daily = daily.drop(columns=["minutes_in_day_local"], errors="ignore")

    # Ensure consistent output schema/order for concatenation with 2019–2022
    if ENFORCE_COLUMN_ORDER:
        for col in OUTPUT_COLUMNS:
            if col not in daily.columns:
                daily[col] = pd.NA
        daily = daily[OUTPUT_COLUMNS]

    # ---- Cheap QA guardrails
    try:
        dupes = daily.duplicated(["fips_code","run_start_time_day"]).sum()
        print("QA: dupes (fips,day) =", dupes)
        print("QA: any_out not in {0,1} =", (~daily["any_out"].isin([0,1]) & daily["any_out"].notna()).sum())
        if minutes:
            off = ((daily["minutes_out"].fillna(0) % minutes) != 0).sum()
            print(f"QA: minutes_out not multiple of {minutes} =", off)
        if "pct_out_max" in daily.columns:
            gt1 = (daily["pct_out_max"] > 1).sum()
            print("QA: pct_out_max > 1 =", gt1)
    except Exception:
        pass

    return daily, minutes

def _warn_on_missing_mapped_fips(df_unique_fips, fips_tz_map, fallback_tz):
    unmapped = sorted(set(df_unique_fips) - set(fips_tz_map.keys()))
    if unmapped:
        print(f"⚠️  {len(unmapped)} FIPS missing explicit TZ mapping. Using FALLBACK_TZ={fallback_tz} for: {', '.join(unmapped)}")

def main():
    # Derive YEAR from filename (best-effort)
    fname = Path(OUTAGE_FILE).name
    year_match = re.search(r"(20\d{2})", fname)
    year = year_match.group(1) if year_match else "unknown"
    out_csv = str(Path(OUT_DIR) / f"outages_daily_{year}.csv")

    # Load MCC (optional)
    mcc = load_mcc(MCC_FILE)

    # Process outages → daily rollups
    daily, minutes = process_outages(
        path=OUTAGE_FILE,
        target_fips=TARGET_FIPS,
        day_mode=DAY_TZ_MODE,
        single_tz=LOCAL_TZ,
        fips_tz_map=FIPS_TZ_MAP,
        fallback_tz=FALLBACK_TZ,
        gap_minutes=GAP_MINUTES,
        mcc=mcc,
        snap_minutes=SNAP_MINUTES,
        chunksize=CHUNKSIZE
    )

    # Save CSV
    write_csv(daily, out_csv)
    print(f"✓ Wrote {out_csv}  rows={len(daily):,}  (snapshot={minutes} min)")

    # Quick QA
    print("\n--- QA: Outages daily ---")
    print("Rows:", len(daily))
    print("FIPS count:", daily['fips_code'].nunique())
    print("Date range:", daily['run_start_time_day'].min(), "→", daily['run_start_time_day'].max())
    for col in ["num_out_per_day","minutes_out","customers_out","pct_out_max","pct_out_area","cust_minute_area"]:
        if col in daily:
            print(f"{col} stats:", daily[col].describe().to_dict())

    _warn_on_missing_mapped_fips(daily["fips_code"].unique().tolist(), FIPS_TZ_MAP, FALLBACK_TZ)

if __name__ == "__main__":
    try:
        pd.options.mode.dtype_backend = "pyarrow" if USE_ARROW_ENGINE else "numpy_nullable"
    except Exception:
        pass
    main()
