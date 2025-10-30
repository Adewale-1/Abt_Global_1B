#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full EDA generator for merged weather/outage dataset (adjusted, bugfix).

Fixes in this version:
- Pandas: use ascending=False (not 'descending') in sort_values for top-corrs.
- Suppress harmless NumPy warnings from correlations on constant columns.
"""

import os
import sys
import math
import json
import argparse
import textwrap
import warnings
from datetime import datetime, timedelta

import numpy as np
np.seterr(divide="ignore", invalid="ignore")  # silence corr/std warnings on constant cols

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# Optional deps
try:
    from statsmodels.tsa.seasonal import STL
    HAS_STL = True
except Exception:
    HAS_STL = False

try:
    from sklearn.feature_selection import mutual_info_classif
    HAS_SK = True
except Exception:
    HAS_SK = False

try:
    from tqdm import tqdm
    TQDM = tqdm
except Exception:
    def TQDM(x, **kwargs): return x

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------- DEFAULT CONFIG ----------------------
DEFAULT_DATA_PATH = r"C:\Users\Owner\.cursor\ABT Global\Abt_Global_1B\data\ml_ready\merged_weather_outages_2019_2024_encoded.csv"

# Outputs
OUTPUT_DIR = r"C:\Users\Owner\.cursor\ABT Global\Abt_Global_1B\results\reports"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")

# Thresholds & analysis knobs
DEFAULT_SEVERITY_THRESHOLDS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
NEAR_ZERO_VAR_FRAC = 0.01  # unique / n < 1%
HIGH_CORR_THRESH = 0.95
EVENT_WINDOW = 7
TOPK_EVENTS = 20   # fallback if quantile yields too few events
RANDOM_STATE = 42
MISSINGNESS_CAL_SKIP_FRAC = 0.005  # skip calendars if <0.5% missing

# Feature families (prefix/contains patterns)
FAMILY_RULES = {
    "precip": ["PRCP", "precip"],
    "wind":   ["WSF", "AWND", "wind"],
    "temp":   ["TMAX", "TMIN", "degree_day", "cooling", "heating", "thermal", "temp_"],
    "flags":  ["WT", "heavy_rain", "extreme_rain", "heat_wave", "extreme_heat", "freezing", "extreme_cold",
               "high_winds", "damaging_winds", "moderate_winds", "strong_winds", "sustained_winds",
               "rapid_wind_increase", "freeze_thaw_cycle", "wet_period_indicator", "wet_windy_combo",
               "multiple_extremes", "winter_freeze", "summer_heat", "spring_storms", "ice_storm_risk",
               "heat_demand_stress", "weather_severity_score"]
}

# Suspected leakage/operational fields
LEAKY_PATTERNS = [
    "coverage", "minutes_observed", "snapshots_count", "minutes_in_local_day",
    "run_start_time_day", "snapshot_minutes"
]
# Targets/derived targets (cannot be features when predicting severity)
TARGET_DERIVED = ["pct_out_max", "pct_out_area", "pct_out_area_unified", "pct_out_area_covered",
                  "pct_out_max_unified", "any_out", "minutes_out", "customers_out",
                  "customers_out_mean", "cust_minute_area"]

# Extra leakage name patterns often seen in engineered/encoded tables
EXTRA_LEAKY = ["label", "target", "outage_label", "y_", "_label", "_target"]

# Columns commonly used across sections
COL_DAY = "day"
COL_FIPS = "fips_code"  # prefer this; we'll fall back to county_fips if needed


# ---------------------- UTILS ----------------------
def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

def to_markdown_table(df: pd.DataFrame, index=False):
    try:
        return df.to_markdown(index=index)
    except Exception:
        buf = df.to_csv(index=index)
        return "```\n" + buf + "\n```"

def write_section(md_lines, title, body_lines):
    md_lines.append(f"## {title}")
    md_lines.extend(body_lines)
    md_lines.append("")

def fmt_bytes(n):
    if n < 1024: return f"{n} B"
    for unit in ['KB','MB','GB','TB']:
        n /= 1024.0
        if n < 1024.0:
            return f"{n:.2f} {unit}"
    return f"{n:.2f} PB"

def present_columns(df):
    return set(df.columns)

def has_cols(df, cols):
    s = present_columns(df)
    return all(c in s for c in cols)

def pick_existing(df, candidates):
    return [c for c in candidates if c in df.columns]

def year_col(df):
    if COL_DAY in df:
        return df[COL_DAY].dt.year
    return pd.Series(np.nan, index=df.index)

def month_str_col(df):
    if COL_DAY in df:
        return df[COL_DAY].dt.strftime("%Y-%m")
    return pd.Series("", index=df.index)

def safe_corr(a: pd.Series, b: pd.Series, method="pearson"):
    s = pd.concat([a, b], axis=1).dropna()
    if s.shape[0] < 3: return np.nan
    return s.iloc[:,0].corr(s.iloc[:,1], method=method)

def group_shift(df, group, col, lag):
    return df.groupby(group)[col].shift(lag)

def save_plot(fname_base):
    path = os.path.join(IMAGES_DIR, fname_base + ".png")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    # paths inside MD must be relative to the MD file location (OUTPUT_DIR)
    return os.path.relpath(path, start=OUTPUT_DIR).replace("\\","/")

def save_table(df, fname_base):
    path = os.path.join(TABLES_DIR, fname_base + ".csv")
    df.to_csv(path, index=False)
    return os.path.relpath(path, start=OUTPUT_DIR).replace("\\","/")

def in_01(series: pd.Series) -> bool:
    try:
        return series.min() >= 0 and series.max() <= 1
    except Exception:
        return False

def pct_label_from_threshold(df: pd.DataFrame, thr: float) -> pd.Series:
    if "pct_out_max" not in df:
        return pd.Series(dtype="float64")
    return (df["pct_out_max"] >= thr).astype(int)


# ---------------------- MAIN EDA ----------------------
def main():
    # ---- CLI ----
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_DATA_PATH, help="Path to input CSV")
    ap.add_argument("--quick", action="store_true", help="Skip heavy sections (STL, missingness calendars)")
    ap.add_argument("--lag_max", type=int, default=3, help="Max absolute lag for temporal cross-correlation (default=3 -> ±3)")
    ap.add_argument("--sev_threshold", type=float, default=0.005, help="Default severity threshold for 'rate' sections")
    args = ap.parse_args()

    DATA_PATH = args.csv
    SEVERITY_THRESHOLDS = DEFAULT_SEVERITY_THRESHOLDS[:]  # already includes 0.05 and 0.10

    ensure_dirs()
    ts = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    report_path = os.path.join(OUTPUT_DIR, f"EDA_Report_{ts}.md")
    md = []

    # Load
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: CSV not found at {DATA_PATH}", file=sys.stderr)
        sys.exit(1)

    # --- Coerce / reconstruct date ---
    def _reconstruct_day(df):
        if "day" in df.columns:
            df["day"] = pd.to_datetime(df["day"], errors="coerce")
            return df
        if "date" in df.columns:
            df["day"] = pd.to_datetime(df["date"], errors="coerce")
            return df
        if {"year", "day_of_year"}.issubset(df.columns):
            y = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
            j = pd.to_numeric(df["day_of_year"], errors="coerce").astype("Int64")
            s = y.astype(str).str.zfill(4) + j.astype(str).str.zfill(3)  # "YYYYDDD"
            df["day"] = pd.to_datetime(s, format="%Y%j", errors="coerce")
            return df
        if {"year", "month"}.issubset(df.columns):
            df["day"] = pd.to_datetime(
                dict(
                    year=pd.to_numeric(df["year"], errors="coerce"),
                    month=pd.to_numeric(df["month"], errors="coerce"),
                    day=15,
                ),
                errors="coerce",
            )
            return df
        return df

    df = pd.read_csv(DATA_PATH)
    df = _reconstruct_day(df)

    # Ensure fips exists and is usable
    if COL_FIPS not in df.columns:
        if "county_fips" in df.columns:
            df[COL_FIPS] = df["county_fips"]
        elif "county_fips_encoded" in df.columns:
            df[COL_FIPS] = df["county_fips_encoded"]
        else:
            df[COL_FIPS] = "UNK"

    # Optional: zero-pad FIPS
    try:
        df[COL_FIPS] = df[COL_FIPS].astype(int).astype(str).str.zfill(5)
    except Exception:
        df[COL_FIPS] = df[COL_FIPS].astype(str)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    object_cols  = df.select_dtypes(include=["object"]).columns.tolist()
    date_cols    = df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()

    # 1) Dataset Overview
    shape = df.shape
    mem_total = df.memory_usage(deep=True).sum()
    date_min = df[COL_DAY].min() if COL_DAY in df else pd.NaT
    date_max = df[COL_DAY].max() if COL_DAY in df else pd.NaT
    county_count = df[COL_FIPS].nunique(dropna=True)
    write_section(md, "1) Dataset Overview", [
        f"- Shape: **{shape[0]:,} rows × {shape[1]:,} columns**",
        f"- Memory usage: **{fmt_bytes(mem_total)}**",
        f"- Date column: `{COL_DAY}` → **{date_min} → {date_max}**" if COL_DAY in df else "- Date column missing",
        f"- County count ({COL_FIPS}): **{county_count}**"
    ])

    # 2) Dtypes Summary
    dtype_counts = df.dtypes.value_counts()
    write_section(md, "2) Dtypes Summary", [
        to_markdown_table(dtype_counts.rename_axis("dtype").reset_index(name="count"))
    ])

    # 3) Missingness
    miss_counts = df.isna().sum()
    with_missing = miss_counts[miss_counts > 0].sort_values(ascending=False)
    write_section(md, "3) Missingness", [
        f"- Columns with missing values: **{(with_missing>0).sum()} / {df.shape[1]}**",
        to_markdown_table(with_missing.head(30).rename("missing_count").reset_index().rename(columns={"index":"column"}))
    ])

    # 4) Duplicates
    dup_exact = int(df.duplicated().sum())
    dup_key = 0
    if has_cols(df, [COL_FIPS, COL_DAY]):
        dup_key = int(df.duplicated(subset=[COL_FIPS, COL_DAY]).sum())
    write_section(md, "4) Duplicates", [
        f"- Exact duplicate rows: **{dup_exact}**",
        f"- Duplicate `({COL_FIPS}, {COL_DAY})` rows: **{dup_key}**"
    ])

    # 5) Date Coverage by County
    if has_cols(df, [COL_FIPS, COL_DAY]):
        g = df.dropna(subset=[COL_DAY]).groupby(COL_FIPS)
        global_min = df[COL_DAY].min()
        global_max = df[COL_DAY].max()
        expected_days = (global_max - global_min).days + 1 if pd.notna(global_min) and pd.notna(global_max) else np.nan
        rows = []
        for fips, d in g:
            first = d[COL_DAY].min()
            last  = d[COL_DAY].max()
            obs_days = d[COL_DAY].nunique()
            miss_est = (expected_days - obs_days) if pd.notna(expected_days) else np.nan
            rows.append([fips, first, last, obs_days, expected_days, miss_est])
        cov_df = pd.DataFrame(rows, columns=[COL_FIPS, "first_date","last_date","observed_days","expected_days","missing_days_est"])
        write_section(md, "5) Date Coverage by County", [to_markdown_table(cov_df)])
    else:
        write_section(md, "5) Date Coverage by County", ["`fips_code` or `day` missing; skipping."])

    # 6) Outage Labels / Diagnostics
    diag_cols = ["any_out","num_out_per_day","minutes_out","customers_out",
                 "cust_minute_area","pct_out_max","pct_out_area","customers_total",
                 "coverage","minutes_in_local_day","snapshots_count","snapshot_minutes",
                 "customers_out_mean","pct_out_area_unified","pct_out_area_covered","pct_out_max_unified"]
    present_diag = pick_existing(df, diag_cols)
    lines = []
    if "any_out" in present_diag:
        val_counts = df["any_out"].value_counts(dropna=False).to_dict()
        pos = val_counts.get(1, 0)
        neg = val_counts.get(0, 0)
        total = pos + neg
        pos_rate = pos / total if total else np.nan
        lines.append(f"- `any_out` distribution: {val_counts} (positives: **{pos_rate*100:.2f}%**)")
    if present_diag:
        stats = df[present_diag].describe().T
        write_section(md, "6) Outage Labels / Diagnostics", lines + [to_markdown_table(stats.reset_index().rename(columns={"index":"feature"}))])
    else:
        write_section(md, "6) Outage Labels / Diagnostics", ["No diagnostic columns found."])

    # 7) Weather Feature Presence
    expected_weather = ["PRCP","TMAX","TMIN","WSF2","AWND","WSF5"]
    presence = {c: (c in df.columns) for c in expected_weather}
    write_section(md, "7) Weather Feature Presence", [f"- Expected weather columns present: {presence}"])

    # 8) High Correlations (|r| ≥ 0.95)
    high_pairs = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 1:
        num_df = df[numeric_cols].copy()
        corr = num_df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        pairs = (upper.stack()
                      .reset_index()
                      .rename(columns={"level_0":"feature_1","level_1":"feature_2",0:"corr"}))
        high_pairs = pairs[pairs["corr"] >= HIGH_CORR_THRESH].sort_values("corr", ascending=False)
        write_section(md, "8) High Correlations (|r| ≥ 0.95)", [
            to_markdown_table(high_pairs.head(50))
        ])
    else:
        write_section(md, "8) High Correlations (|r| ≥ 0.95)", ["Insufficient numeric columns."])

    # 9) Feature–Target Correlations (top |r|) [Pearson & Spearman]
    def top_corrs_to_target(target, k=20, method="pearson"):
        if target not in df.columns: return pd.DataFrame()
        cols = [c for c in numeric_cols if c != target]
        out = []
        for c in cols:
            r = safe_corr(df[c], df[target], method=method)
            out.append((c, r))
        res = pd.DataFrame(out, columns=["feature","corr"]).dropna()
        res["abs"] = res["corr"].abs()
        return res.sort_values("abs", ascending=False).drop(columns=["abs"]).head(k)

    for tgt in ["any_out","minutes_out","customers_out","pct_out_max"]:
        if tgt not in df.columns:
            write_section(md, f"9) Feature–Target Correlations — target: `{tgt}`", [f"`{tgt}` not found; skipping."])
            continue
        pear = top_corrs_to_target(tgt, k=20, method="pearson")
        spear = top_corrs_to_target(tgt, k=20, method="spearman")
        if pear.empty and spear.empty:
            write_section(md, f"9) Feature–Target Correlations — target: `{tgt}`", ["No numeric features or insufficient data."])
        else:
            write_section(md, f"9) Feature–Target Correlations — target: `{tgt}` (Pearson)", [to_markdown_table(pear)])
            write_section(md, f"9) Feature–Target Correlations — target: `{tgt}` (Spearman)", [to_markdown_table(spear)])

    # 10) Constant & Near-Zero-Variance
    nzv_rows = []
    const_cols = []
    n = len(df)
    for c in numeric_cols + object_cols:
        u = df[c].nunique(dropna=True)
        if u <= 1:
            const_cols.append(c)
        frac = u / max(n, 1)
        if frac < NEAR_ZERO_VAR_FRAC:
            nzv_rows.append((c, u, frac))
    nzv_df = pd.DataFrame(nzv_rows, columns=["column","unique_count","unique_frac"]).sort_values("unique_frac")
    write_section(md, "10) Constant & Near-Zero-Variance", [
        f"- Constant columns: **{len(const_cols)}**",
        f"- Near-zero-variance (unique_frac < {NEAR_ZERO_VAR_FRAC:.0%}): **{len(nzv_df)}**",
        to_markdown_table(nzv_df.head(60))
    ])

    # 11) Skewness & IQR Outliers (numeric only)
    skew = df[numeric_cols].skew(numeric_only=True).sort_values(ascending=False)
    outlier_counts = []
    for c in numeric_cols:
        s = df[c].dropna()
        if s.empty:
            outlier_counts.append((c, 0))
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr <= 0:
            outlier_counts.append((c, 0))
            continue
        lower = q1 - 1.5*iqr
        upper = q3 + 1.5*iqr
        oc = int(((s < lower) | (s > upper)).sum())
        outlier_counts.append((c, oc))
    outlier_df = pd.DataFrame(outlier_counts, columns=["feature","outlier_count"]).sort_values("outlier_count", ascending=False)
    write_section(md, "11) Skewness & IQR Outliers (numeric only)", [
        "Top skewness:",
        to_markdown_table(skew.head(30).rename("skew").reset_index().rename(columns={"index":"feature"})),
        "",
        "Outlier counts by IQR rule (top 50):",
        to_markdown_table(outlier_df.head(50))
    ])

    # 12) Memory Usage
    mem = df.memory_usage(deep=True).sort_values(ascending=False)
    write_section(md, "12) Memory Usage", [
        f"- Total: **{fmt_bytes(mem_total)}**",
        to_markdown_table(mem.rename("bytes").reset_index().rename(columns={"index":"column"}))
    ])

    # 13) Per-County, Per-Month Severe-Outage Rate (thresholded pct_out_max)
    if COL_DAY in df:
        if "pct_out_max" in df.columns:
            thr = args.sev_threshold
            temp = df.copy()
            temp["_month"] = month_str_col(temp)
            temp["sev_label"] = (temp["pct_out_max"] >= thr).astype(int)
            grp = temp.groupby([COL_FIPS, "_month"])["sev_label"].mean().reset_index(name=f"rate_ge_{thr}")
            cnt = temp.groupby([COL_FIPS, "_month"])["sev_label"].size().reset_index(name="n")
            per = grp.merge(cnt, on=[COL_FIPS,"_month"])
            write_section(md, f"13) Per-County, Per-Month Severe-Outage Rate (pct_out_max ≥ {thr})", [to_markdown_table(per.head(200))])
        else:
            write_section(md, "13) Per-County, Per-Month Severe-Outage Rate", ["`pct_out_max` missing; skipping."])
    else:
        write_section(md, "13) Per-County, Per-Month Severe-Outage Rate", ["No date column."])

    # 14) Outage Severity Threshold Sweep (pct_out_max)
    if "pct_out_max" in df.columns:
        rows = []
        for thr in DEFAULT_SEVERITY_THRESHOLDS:
            prev = float((df["pct_out_max"] >= thr).mean())
            rows.append((thr, prev, len(df)))
        sweep = pd.DataFrame(rows, columns=["pct_thr","prevalence","n"])
        write_section(md, "14) Outage Severity Threshold Sweep (pct_out_max)", [to_markdown_table(sweep)])
    else:
        write_section(md, "14) Outage Severity Threshold Sweep (pct_out_max)", ["`pct_out_max` not found; skipping."])

    # 15) Temporal Cross-Correlation (±lag_max) — Pearson & Spearman
    xcorr_rows = []
    if has_cols(df, [COL_DAY, COL_FIPS]):
        candidates = pick_existing(df, ["PRCP_7d_sum","PRCP","TMAX","WSF2","WSF2_7d_max","AWND"])
        target = None
        if "minutes_out" in df.columns:
            target = "minutes_out"
        elif "pct_out_max" in df.columns:
            target = "pct_out_max"
        if target and candidates:
            temp = df[[COL_FIPS, COL_DAY, target] + candidates].dropna(subset=[COL_DAY]).copy()
            temp = temp.sort_values([COL_FIPS, COL_DAY])
            for feat in candidates:
                for lag in range(-args.lag_max, args.lag_max + 1):
                    temp[f"{feat}_lag{lag}"] = group_shift(temp, COL_FIPS, feat, lag)
                    for method in ("pearson", "spearman"):
                        r = safe_corr(temp[f"{feat}_lag{lag}"], temp[target], method=method)
                        xcorr_rows.append((feat, target, lag, method, r))
            xcorr_df = pd.DataFrame(xcorr_rows, columns=["feature","target","lag","method","corr"]).sort_values(["feature","method","lag"])
            write_section(md, "15) Temporal Cross-Correlation (±lag)", [to_markdown_table(xcorr_df.head(400))])
            save_table(xcorr_df, "temporal_xcorr")
        else:
            write_section(md, "15) Temporal Cross-Correlation (±lag)", ["Missing target or candidate features; skipping."])
    else:
        write_section(md, "15) Temporal Cross-Correlation (±lag)", ["Need day and fips columns."])

    # 16) Integrity / Consistency Checks (encoded-aware)
    checks = {}
    if has_cols(df, ["customers_out","customers_total"]):
        checks["customers_out_leq_total_viol"] = int((df["customers_out"] > df["customers_total"]).sum())
    if "pct_out_max" in df.columns:
        checks["pct_out_max_outside_0_1"] = int(((df["pct_out_max"] < 0) | (df["pct_out_max"] > 1)).sum())

    if "coverage" in df.columns and "minutes_out" in df.columns:
        # minutes_in_local_day may be missing; use 1440 fallback and clip to [1380,1500]
        if "minutes_in_local_day" in df.columns:
            mid = pd.to_numeric(df["minutes_in_local_day"], errors="coerce").fillna(1440).clip(1380, 1500)
        else:
            mid = pd.Series(1440, index=df.index)

        if in_01(pd.to_numeric(df["coverage"], errors="coerce")):
            checks["minutes_out_mismatch_fullcov"] = int(((pd.to_numeric(df["coverage"], errors="coerce") == 1) &
                                                          (pd.to_numeric(df["minutes_out"], errors="coerce") > mid)).sum())
        else:
            checks["minutes_out_mismatch_fullcov"] = "skipped (coverage encoded or not in [0,1])"
    write_section(md, "16) Integrity / Consistency Checks", [to_markdown_table(pd.DataFrame(list(checks.items()), columns=["check","count"]))])

    # 17) Identical Numeric Columns
    ident_rows = []
    try:
        hashes = {c: pd.util.hash_pandas_object(df[c], index=False).sum() for c in numeric_cols}
        inv = {}
        for c, h in hashes.items():
            inv.setdefault(h, []).append(c)
        groups = [v for v in inv.values() if len(v) > 1]
        if groups:
            for gcols in groups:
                ident_rows.append({"group": ", ".join(gcols)})
            write_section(md, "17) Identical Numeric Columns", [to_markdown_table(pd.DataFrame(ident_rows))])
        else:
            write_section(md, "17) Identical Numeric Columns", ["None"])
    except Exception as e:
        write_section(md, "17) Identical Numeric Columns", [f"Error computing: {e}"])

    # 18) Year-over-Year Means (selected)
    sel_cols = pick_existing(df, ["PRCP","TMAX","minutes_out","pct_out_max","customers_out"])
    if COL_DAY in df and sel_cols:
        tmp = df[[COL_DAY] + sel_cols].copy()
        tmp["_year"] = tmp[COL_DAY].dt.year
        yoy = tmp.groupby("_year")[sel_cols].mean().reset_index()
        write_section(md, "18) Year-over-Year Means (selected)", [to_markdown_table(yoy)])
    else:
        write_section(md, "18) Year-over-Year Means (selected)", ["Missing date or selected columns."])

    # 19) Flag Hierarchy Checks
    rows = []
    if has_cols(df, ["extreme_rain","heavy_rain"]):
        viol = ((df["extreme_rain"] == 1) & (df["heavy_rain"] != 1)).sum()
        rows.append(("extreme_implies_heavy_viol", int(viol)))
    write_section(md, "19) Flag Hierarchy Checks", [to_markdown_table(pd.DataFrame(rows, columns=["check","count"]))] if rows else ["No applicable flag pairs found."])

    # 20) Missingness by County × Month (long)
    if COL_DAY in df:
        key_cols = pick_existing(df, ["AWND","WSF2","PRCP"])
        if key_cols:
            overall_missing = float(pd.concat([df[k] for k in key_cols], axis=1).isna().mean().mean())
            if overall_missing < MISSINGNESS_CAL_SKIP_FRAC:
                write_section(md, "20) Missingness by County × Month (long)", [f"All key sensors have <{100*MISSINGNESS_CAL_SKIP_FRAC:.1f}% missing; skipped."])
            else:
                tmp = df[[COL_FIPS, COL_DAY] + key_cols].copy()
                tmp["_month"] = tmp[COL_DAY].dt.to_period("M").astype(str)
                long = tmp.melt(id_vars=[COL_FIPS,"_month"], value_vars=key_cols, var_name="column", value_name="val")
                long["is_missing"] = long["val"].isna().astype(int)
                miss = (long.groupby([COL_FIPS,"_month","column"])["is_missing"]
                            .mean().reset_index(name="missing_frac"))
                write_section(md, "20) Missingness by County × Month (long)", [to_markdown_table(miss.head(200))])
        else:
            write_section(md, "20) Missingness by County × Month (long)", ["No AWND/WSF2/PRCP present."])
    else:
        write_section(md, "20) Missingness by County × Month (long)", ["No date column."])

    # 21) Fold Planning Summary (year × county) using severe label
    if COL_DAY in df:
        if "pct_out_max" in df.columns:
            thr = args.sev_threshold
            tmp = df[[COL_FIPS, COL_DAY, "pct_out_max"]].copy()
            tmp["sev_label"] = (tmp["pct_out_max"] >= thr).astype(int)
            tmp["_year"] = tmp[COL_DAY].dt.year
            g = tmp.groupby(["_year", COL_FIPS])
            fold = g["sev_label"].agg(n="size", any_out_rate="mean").reset_index()
            write_section(md, f"21) Fold Planning Summary (year × county) — label: pct_out_max ≥ {thr}", [to_markdown_table(fold)])
        else:
            write_section(md, "21) Fold Planning Summary (year × county)", ["No outage-like target found."])
    else:
        write_section(md, "21) Fold Planning Summary (year × county)", ["No date column."])

    # 22) Target Definition & Severity by County/Year
    if "pct_out_max" in df.columns and COL_DAY in df:
        tmp = df[[COL_FIPS, COL_DAY, "pct_out_max"]].copy()
        tmp["_year"] = tmp[COL_DAY].dt.year
        all_rows = []
        for thr in DEFAULT_SEVERITY_THRESHOLDS:
            tmp[f"label_{thr}"] = (tmp["pct_out_max"] >= thr).astype(int)
            grp = tmp.groupby(["_year", COL_FIPS])[f"label_{thr}"].agg(prevalence="mean", n="size").reset_index()
            grp["threshold"] = thr
            all_rows.append(grp)
        res = pd.concat(all_rows, ignore_index=True)
        write_section(md, "22) Target Definition & Severity by County/Year", [to_markdown_table(res.head(200))])
        save_table(res, "severity_by_county_year")
    else:
        write_section(md, "22) Target Definition & Severity by County/Year", ["Need `pct_out_max` and date column."])

    # 23) Baseline & Class-Imbalance Benchmarks (severity labels)
    if "pct_out_max" in df.columns and COL_DAY in df:
        tmp = df[[COL_FIPS, COL_DAY, "pct_out_max"]].copy()
        tmp["_year"] = tmp[COL_DAY].dt.year
        rows = []
        for thr in DEFAULT_SEVERITY_THRESHOLDS:
            tmp["label"] = (tmp["pct_out_max"] >= thr).astype(int)
            grp = tmp.groupby([COL_FIPS, "_year"])["label"].agg(["mean","size"]).reset_index()
            grp = grp.rename(columns={"mean":"prevalence","size":"n"})
            grp["majority_acc"] = grp["prevalence"].apply(lambda p: max(p, 1-p) if 0 < p < 1 else 1.0 if p in [0,1] else np.nan)
            grp["auroc_baseline"] = grp["prevalence"].apply(lambda p: 0.5 if 0 < p < 1 else np.nan)
            grp["auprc_baseline"] = grp["prevalence"]
            grp["threshold"] = thr
            rows.append(grp)
        base_df = pd.concat(rows, ignore_index=True)
        write_section(md, "23) Baseline & Class-Imbalance Benchmarks", [to_markdown_table(base_df.head(200))])
        save_table(base_df, "baseline_class_imbalance")
    else:
        write_section(md, "23) Baseline & Class-Imbalance Benchmarks", ["Need `pct_out_max` and date column."])

    # 24) Seasonality & Trend (STL) per County
    if args.quick:
        write_section(md, "24) Seasonality & Trend (STL) per County", ["Skipped (--quick)."])
    elif not HAS_STL:
        write_section(md, "24) Seasonality & Trend (STL) per County", ["`statsmodels` not installed; skipping."])
    elif "pct_out_max" in df.columns and COL_DAY in df:
        notes = []
        for fips, d in TQDM(df[[COL_FIPS, COL_DAY, "pct_out_max"]].dropna(subset=[COL_DAY]).groupby(COL_FIPS)):
            s = d.set_index(COL_DAY)["pct_out_max"].sort_index().asfreq("D").fillna(0.0)
            if len(s) < 400:
                continue
            try:
                stl = STL(s, period=365, robust=True).fit()
                plt.figure(figsize=(8,5))
                plt.plot(stl.trend, label="trend")
                plt.plot(stl.seasonal, label="seasonal", alpha=0.7)
                plt.plot(stl.resid, label="resid", alpha=0.5)
                plt.title(f"STL components — fips {fips}")
                plt.legend()
                rel = save_plot(f"stl_fips_{fips}")
                notes.append(f"- fips {fips}: ![stl](./{rel})")
            except Exception:
                continue
        write_section(md, "24) Seasonality & Trend (STL) per County", notes if notes else ["Insufficient data for STL."])
    else:
        write_section(md, "24) Seasonality & Trend (STL) per County", ["Need `pct_out_max` and date column."])

    # 25) Event-Window Response (±7 days), quantile-based events, broader drivers
    drivers = pick_existing(df, ["PRCP","WSF2","AWND","high_winds","damaging_winds","rapid_wind_increase","wet_windy_combo"])
    metrics = pick_existing(df, ["pct_out_max","minutes_out","PRCP","WSF2","AWND"])
    if COL_DAY in df and drivers and metrics:
        notes = []
        for drv in drivers:
            all_windows = []

            # Build a unique column list to avoid duplicate names creating DataFrame slices
            cols = [COL_FIPS, COL_DAY] + metrics + [drv]
            cols = list(dict.fromkeys(cols))  # remove duplicates, keep order

            grouped = df[cols].dropna(subset=[COL_DAY]).groupby(COL_FIPS)
            for fips, d in TQDM(grouped):
                dd = d.sort_values(COL_DAY).set_index(COL_DAY)

                # Use the first matching column name (in case of duplicates)
                drv_matches = [c for c in dd.columns if c == drv]
                if not drv_matches:
                    continue

                drv_series = pd.to_numeric(dd[drv_matches[0]], errors="coerce")
                if drv_series.dropna().empty:
                    continue

                # Quantile-based event selection: continuous -> >= 95th percentile; binary -> value==1
                event_idx = []
                if drv_series.dropna().isin([0,1]).all():
                    event_idx = drv_series[drv_series == 1].index
                else:
                    thr_q = drv_series.quantile(0.95)
                    event_idx = drv_series[drv_series >= thr_q].index

                # Fallback: if too few events, take top-K (guard for constant/flat series)
                if len(event_idx) < 3:
                    try:
                        event_idx = drv_series.nlargest(min(TOPK_EVENTS, max(1, drv_series.shape[0] // 30))).index
                    except Exception:
                        event_idx = drv_series.dropna().sort_values().tail(min(TOPK_EVENTS, max(1, drv_series.shape[0] // 30))).index

                if len(event_idx) == 0:
                    continue

                for t0 in event_idx:
                    for lag in range(-EVENT_WINDOW, EVENT_WINDOW + 1):
                        ts = t0 + timedelta(days=lag)
                        if ts in dd.index:
                            row = {"fips": fips, "lag": lag}
                            for m in metrics:
                                mm = [c for c in dd.columns if c == m]
                                if not mm:
                                    continue
                                row[m] = pd.to_numeric(dd.loc[ts, mm[0]], errors="coerce")
                            all_windows.append(row)

            if all_windows:
                win_df = pd.DataFrame(all_windows)
                agg = win_df.groupby("lag")[metrics].mean(numeric_only=True).reset_index()

                for m in metrics:
                    if m not in agg.columns:
                        continue
                    plt.figure(figsize=(7, 4))
                    plt.plot(agg["lag"], agg[m])
                    plt.axvline(0, linestyle="--")
                    plt.title(f"Event-window avg — align on top {drv} days — metric: {m}")
                    rel = save_plot(f"eventwin_{drv}_{m}")
                    notes.append(f"- {drv} → {m}: ![plot](./{rel})")

                save_table(agg, f"eventwindow_{drv}_agg")

        write_section(md, "25) Event-Window Response (±7 days)", notes if notes else ["No sufficient data to build windows."])
    else:
        write_section(md, "25) Event-Window Response (±7 days)", ["Need day, drivers, and metrics."])

    # 26) Feature Family Pruning Plan
    fam_notes = []
    kept_dropped_rows = []
    for fam, patterns in FAMILY_RULES.items():
        fam_cols = [c for c in numeric_cols if any(p.lower() in c.lower() for p in patterns)]
        fam_cols = [c for c in fam_cols if c not in TARGET_DERIVED]
        if len(fam_cols) < 2:
            continue
        sub = df[fam_cols].copy().dropna(axis=1, how="all").dropna(axis=0, how="all")
        if sub.shape[1] < 2:
            continue
        corr = sub.corr().abs()
        plt.figure(figsize=(max(6, min(12, 0.4*len(fam_cols))), 6))
        plt.imshow(corr, aspect='auto')
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=6)
        plt.yticks(range(len(corr.index)), corr.index, fontsize=6)
        plt.colorbar()
        plt.title(f"{fam} family correlation")
        rel = save_plot(f"family_corr_{fam}")
        fam_notes.append(f"- {fam} corr heatmap: ![heatmap](./{rel})")
        kept, dropped = [], []
        seen = set()
        for c in corr.columns:
            if c in seen:
                continue
            kept.append(c)
            seen.add(c)
            high = corr.index[(corr[c] >= HIGH_CORR_THRESH) & (corr.index != c)].tolist()
            for h in high:
                if h not in seen:
                    dropped.append(h)
                    seen.add(h)
        kept_dropped_rows.append({"family": fam, "kept": ", ".join(kept), "dropped_candidates": ", ". join(sorted(set(dropped)))})
    kd_df = pd.DataFrame(kept_dropped_rows)
    write_section(md, "26) Feature Family Pruning Plan", fam_notes + ([to_markdown_table(kd_df)] if not kd_df.empty else ["No families pruned."]))

    # 27) Leakage Audit & Exclusion List (broadened)
    leak_rows = []
    for c in df.columns:
        reason = None
        cl = c.lower()
        if any(p in cl for p in [p.lower() for p in LEAKY_PATTERNS]):
            reason = "Operational/measurement timing"
        if reason is None and any(p in cl for p in [p.lower() for p in TARGET_DERIVED]):
            reason = "Target/derived-from-target"
        if reason is None and any(p in cl for p in [p.lower() for p in EXTRA_LEAKY]):
            reason = "Engineered/explicit target label"
        if reason:
            leak_rows.append({"column": c, "reason": reason})
    write_section(md, "27) Leakage Audit & Exclusion List",
                  [to_markdown_table(pd.DataFrame(leak_rows))] if leak_rows else ["No suspected leakage fields found."])

    # 28) Missingness Calendars (skip if near-complete or --quick)
    if args.quick:
        write_section(md, "28) Missingness Calendars", ["Skipped (--quick)."])
    elif COL_DAY in df:
        sensors = pick_existing(df, ["AWND","WSF2","PRCP"])
        if sensors:
            overall_missing = float(pd.concat([df[k] for k in sensors], axis=1).isna().mean().mean())
            if overall_missing < MISSINGNESS_CAL_SKIP_FRAC:
                write_section(md, "28) Missingness Calendars", [f"All key sensors have <{100*MISSINGNESS_CAL_SKIP_FRAC:.1f}% missing; skipped."])
            else:
                cal_notes = []
                for sensor in sensors:
                    for fips, d in TQDM(df[[COL_FIPS, COL_DAY, sensor]].dropna(subset=[COL_DAY]).groupby(COL_FIPS)):
                        s = d.set_index(COL_DAY)[sensor]
                        cal = []
                        s = s.asfreq("D")
                        if s.index.min() is None or s.index.max() is None:
                            continue
                        months = pd.period_range(s.index.min().to_period("M"), s.index.max().to_period("M"), freq="M")
                        for m in months:
                            month_dates = pd.date_range(m.start_time, m.end_time, freq="D")
                            vals = s.reindex(month_dates)
                            miss = vals.isna().astype(int)
                            row = {"_month": str(m)}
                            for dom in range(1, 32):
                                row[f"d{dom:02d}"] = miss.iloc[dom-1] if dom <= len(month_dates) else np.nan
                            cal.append(row)
                        cal_df = pd.DataFrame(cal)
                        if cal_df.empty:
                            continue
                        arr = cal_df[[f"d{d:02d}" for d in range(1,32)]].values
                        plt.figure(figsize=(9, max(3, 0.25*len(cal_df))))
                        plt.imshow(arr, aspect="auto", interpolation="nearest")
                        plt.yticks(range(len(cal_df)), cal_df["_month"].tolist(), fontsize=6)
                        plt.xticks(range(31), [str(d) for d in range(1,32)], fontsize=6, rotation=90)
                        plt.colorbar(label="Missing (1) / Present (0)")
                        plt.title(f"Missingness calendar — {sensor} — fips {fips}")
                        rel = save_plot(f"missing_cal_{sensor}_{fips}")
                        cal_notes.append(f"- {sensor}, fips {fips}: ![cal](./{rel})")
                write_section(md, "28) Missingness Calendars", cal_notes if cal_notes else ["Insufficient data to render calendars."])
        else:
            write_section(md, "28) Missingness Calendars", ["No AWND/WSF2/PRCP present."])
    else:
        write_section(md, "28) Missingness Calendars", ["No date column."])

    # 29) Drift & Shift Checks
    if COL_DAY in df:
        key_feats = pick_existing(df, ["PRCP","WSF2","TMAX","pct_out_max"])
        tmp = df[[COL_DAY] + key_feats].copy()
        tmp["_year"] = tmp[COL_DAY].dt.year
        agg = tmp.groupby("_year")[key_feats].mean().reset_index()
        if "pct_out_max" in df.columns:
            sev = []
            for thr in DEFAULT_SEVERITY_THRESHOLDS:
                lab = (df["pct_out_max"] >= thr).astype(int)
                s = pd.DataFrame({COL_DAY: df[COL_DAY], "label": lab})
                s["_year"] = s[COL_DAY].dt.year
                sv = s.groupby("_year")["label"].mean().reset_index()
                sv["threshold"] = thr
                sev.append(sv)
            sev_df = pd.concat(sev, ignore_index=True)
            sev_tab = sev_df.pivot(index="_year", columns="threshold", values="label").reset_index()
            write_section(md, "29) Drift & Shift Checks", [
                "Year-over-year feature means:",
                to_markdown_table(agg),
                "",
                "Year-over-year severe-target prevalence:",
                to_markdown_table(sev_tab)
            ])
        else:
            write_section(md, "29) Drift & Shift Checks", ["`pct_out_max` missing; only feature means shown.", to_markdown_table(agg)])
    else:
        write_section(md, "29) Drift & Shift Checks", ["No date column."])

    # 30) Interactions & Nonlinearities Quick-Scan (MI)
    notes = []
    if "pct_out_max" in df.columns:
        y = (df["pct_out_max"] >= args.sev_threshold).astype(int)
        cand_single = pick_existing(df, ["WSF2_7d_max","PRCP_3d_sum","WSF2","PRCP","TMAX_7d_mean","TMIN_7d_mean","AWND"])
        cand_pairs = [("WSF2_7d_max","PRCP_3d_sum"), ("WSF2","PRCP"), ("TMAX_7d_mean","WSF2_7d_max")]
        cand_pairs = [(a,b) for (a,b) in cand_pairs if a in df.columns and b in df.columns]
        res_rows = []
        if HAS_SK and cand_single:
            Xs = df[cand_single].copy()
            Xs = Xs.fillna(Xs.median())
            try:
                mi_single = mutual_info_classif(Xs, y, random_state=RANDOM_STATE, discrete_features=False)
                for f, mi in zip(cand_single, mi_single):
                    res_rows.append(("single", f, "", float(mi)))
            except Exception:
                notes.append("- sklearn mutual_info_classif failed on singles; skipped.")
        else:
            notes.append("- scikit-learn not installed or no candidate singles; skipping MI for singles.")

        if HAS_SK and cand_pairs:
            for (a,b) in cand_pairs:
                z = (df[a] * df[b]).to_frame(name=f"{a}*{b}")
                z = z.fillna(z.median())
                try:
                    mi = mutual_info_classif(z, y, random_state=RANDOM_STATE, discrete_features=False)[0]
                    res_rows.append(("interaction", a, b, float(mi)))
                except Exception:
                    continue
        else:
            notes.append("- scikit-learn not installed or no candidate pairs; skipping MI for interactions.")

        if res_rows:
            inter_df = pd.DataFrame(res_rows, columns=["type","feature_a","feature_b","mutual_info"]).sort_values("mutual_info", ascending=False)
            write_section(md, "30) Interactions & Nonlinearities Quick-Scan", [to_markdown_table(inter_df)])
        else:
            write_section(md, "30) Interactions & Nonlinearities Quick-Scan", notes if notes else ["No results."])
    else:
        write_section(md, "30) Interactions & Nonlinearities Quick-Scan", ["`pct_out_max` missing."])

    # ------------- FINAL WRITE -------------
    header = [
        f"# EDA Report — merged_weather_outages_2019_2024_encoded (adjusted)",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- Source: `{DATA_PATH}`",
        f"- Args: quick={args.quick}, lag_max=±{args.lag_max}, sev_threshold={args.sev_threshold}",
        ""
    ]
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(header + md))

    print(f"✅ EDA complete.\nMarkdown: {report_path}\nAssets: {OUTPUT_DIR}\\(images|tables)")


if __name__ == "__main__":
    main()
