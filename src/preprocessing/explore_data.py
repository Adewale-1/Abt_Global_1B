# src/preprocessing/explore_data.py
from __future__ import annotations
import argparse, glob, os, sys
from typing import Optional, Tuple, List
from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# ---- Defaults ----
DEFAULT_MERGED_PATH = r"C:\Users\Owner\.cursor\ABT Global\Abt_Global_1B\src\data_collection\data\processed\merged_weather_outages_2019_2024_keep_all.csv"
ML_READY_PATH = "src/data_collection/data/ml_ready/power_outage_ml_features.csv"
PROCESSED_UNIFIED_PATH = "src/data_collection/data/processed/unified_weather_features_5year.csv"
PROCESSED_FALLBACK_GLOB = "src/data_collection/data/processed/weather_features_FIPS_*.csv"

def resolve_input(dataset: str, path: Optional[str]) -> Tuple[str, Optional[str]]:
    if path:
        return path, None
    ds = dataset.lower()
    if ds == "merged_default":
        return DEFAULT_MERGED_PATH, None
    if ds == "ml_ready":
        return ML_READY_PATH, None
    if ds == "processed":
        return PROCESSED_UNIFIED_PATH, PROCESSED_FALLBACK_GLOB
    if ds == "auto":
        if os.path.exists(DEFAULT_MERGED_PATH):
            return DEFAULT_MERGED_PATH, None
        if os.path.exists(ML_READY_PATH):
            return ML_READY_PATH, None
        if os.path.exists(PROCESSED_UNIFIED_PATH):
            return PROCESSED_UNIFIED_PATH, None
        return PROCESSED_UNIFIED_PATH, PROCESSED_FALLBACK_GLOB
    raise ValueError("Unknown --dataset. Use merged_default | ml_ready | processed | auto.")

def load_dataset(primary: str, fallback_glob: Optional[str]) -> pd.DataFrame:
    if os.path.exists(primary):
        print(f"[info] Loading dataset: {primary}")
        return pd.read_csv(primary, low_memory=False)
    if fallback_glob is None:
        raise FileNotFoundError(f"File not found: {primary}")
    files = sorted(glob.glob(fallback_glob))
    if not files:
        raise FileNotFoundError(f"Neither file nor fallback found.\n  primary: {primary}\n  glob: {fallback_glob}")
    print(f"[info] Primary not found. Concatenating {len(files)} files from: {fallback_glob}")
    dfs = [pd.read_csv(f, low_memory=False) for f in files]
    return pd.concat(dfs, ignore_index=True)

def pick_date_column(df: pd.DataFrame, preferred: Optional[str]) -> Optional[str]:
    if preferred and preferred in df.columns: 
        return preferred
    for cand in ("day","date","observation_date","run_start_time_day","DATE","Date"):
        if cand in df.columns: 
            return cand
    return None

def _safe_pct(n: int, d: int) -> float:
    return (float(n) / float(d) * 100.0) if d else 0.0

def _top_abs_corr(series: pd.Series, frame: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    """Pearson corr of a numeric/0-1 label against numeric columns; returns top |r|."""
    try:
        num = frame.select_dtypes(include="number")
        if series.name not in num.columns:
            tmp = num.assign(_target_=series.values)
            target_col = "_target_"
        else:
            tmp = num.copy()
            target_col = series.name
        corrs = tmp.corr(numeric_only=True)[target_col].drop(target_col, errors="ignore").dropna()
        out = corrs.reindex(corrs.abs().sort_values(ascending=False).index).head(k)
        return out.reset_index(names="feature").rename(columns={target_col:"corr"})
    except Exception:
        return pd.DataFrame(columns=["feature","corr"])

def _write_artifact_csv(df: pd.DataFrame, path: str) -> None:
    try:
        df.to_csv(path, index=False)
        print(f"[artifact] wrote {path}")
    except Exception as e:
        print(f"[warn] could not write artifact {path}: {e}")

def _xcorr_lags(x: pd.Series, y: pd.Series, max_lag: int = 2) -> pd.DataFrame:
    out = []
    sx = pd.Series(x).astype(float)
    sy = pd.Series(y).astype(float)
    for lag in range(-max_lag, max_lag+1):
        corr = sx.shift(lag).corr(sy)
        out.append({"lag": lag, "corr": corr})
    return pd.DataFrame(out)

def main():
    load_dotenv()

    p = argparse.ArgumentParser(description="Generate a full Markdown EDA report + CSV artifacts for the dataset.")
    p.add_argument("--dataset",
                   choices=["merged_default","ml_ready","processed","auto"],
                   default="merged_default",
                   help="Select a known dataset (default: merged_default).")
    p.add_argument("--path", default=None, help="Explicit CSV path (overrides --dataset).")
    p.add_argument("--date-col", default=None, help="Date column name (auto-detected if omitted).")
    p.add_argument("--corr-threshold", type=float, default=0.95,
                   help="Absolute correlation threshold for listing high-corr pairs (default 0.95).")
    p.add_argument("--out", default=None,
                   help="Output Markdown report path. Default: alongside input as *_EDA_<timestamp>.md")
    p.add_argument("--no-artifacts", action="store_true",
                   help="Do not write CSV artifacts (coverage, high-corr, label-dist, etc.).")
    args = p.parse_args()

    primary, fb = resolve_input(args.dataset, args.path)
    df = load_dataset(primary, fb)

    # Normalize keys if present
    if "fips_code" in df.columns:
        df["fips_code"] = df["fips_code"].astype(str).str[-5:].str.zfill(5)

    date_col = pick_date_column(df, args.date_col)
    if date_col:
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        except Exception as e:
            print(f"[warn] Could not convert {date_col} to datetime: {e}")

    # Prepare output paths
    in_dir = os.path.dirname(primary)
    in_stem = os.path.splitext(os.path.basename(primary))[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = args.out or os.path.join(in_dir, f"{in_stem}_EDA_{ts}.md")

    # Artifacts
    art_coverage  = os.path.join(in_dir, f"{in_stem}_coverage_by_county_{ts}.csv")
    art_highcorr  = os.path.join(in_dir, f"{in_stem}_high_corr_pairs_{ts}.csv")
    art_labeldist = os.path.join(in_dir, f"{in_stem}_label_distribution_{ts}.csv")
    art_bycnty_mo = os.path.join(in_dir, f"{in_stem}_any_out_by_county_month_{ts}.csv")
    art_severity  = os.path.join(in_dir, f"{in_stem}_severity_threshold_sweep_{ts}.csv")
    art_xcorr     = os.path.join(in_dir, f"{in_stem}_temporal_xcorr_lags_{ts}.csv")
    art_integrity = os.path.join(in_dir, f"{in_stem}_integrity_checks_{ts}.csv")
    art_identical = os.path.join(in_dir, f"{in_stem}_identical_num_cols_{ts}.csv")
    art_yearmeans = os.path.join(in_dir, f"{in_stem}_yearly_means_{ts}.csv")
    art_flags     = os.path.join(in_dir, f"{in_stem}_flag_hierarchy_violations_{ts}.csv")
    art_topcorr_y = os.path.join(in_dir, f"{in_stem}_top_corr_targets_{ts}.csv")
    art_miss_ctym = os.path.join(in_dir, f"{in_stem}_missing_by_county_month_long_{ts}.csv")
    art_folds     = os.path.join(in_dir, f"{in_stem}_fold_planning_{ts}.csv")

    # Begin report
    lines: List[str] = []
    lines.append(f"# EDA Report — {in_stem}")
    lines.append(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Source: `{primary}`")
    lines.append("")

    # Overview
    lines.append("## 1) Dataset Overview")
    lines.append(f"- Shape: **{df.shape[0]:,} rows × {df.shape[1]} columns**")
    mem_mb = df.memory_usage(deep=True).sum() / (1024**2)
    lines.append(f"- Memory usage: **{mem_mb:.2f} MB**")
    if date_col:
        date_min, date_max = df[date_col].min(), df[date_col].max()
        lines.append(f"- Date column: `{date_col}` → **{date_min} → {date_max}**")
    if "fips_code" in df.columns:
        lines.append(f"- County count (fips_code): **{df['fips_code'].nunique()}**")
    lines.append("")

    # Dtypes summary
    lines.append("## 2) Dtypes Summary")
    dtype_counts = df.dtypes.value_counts().to_dict()
    lines.append(f"- Dtype counts: {dtype_counts}")
    lines.append("")

    # Missingness
    lines.append("## 3) Missingness")
    miss = df.isnull().sum().sort_values(ascending=False)
    top_miss = miss[miss > 0].head(30)
    lines.append(f"- Columns with missing values: **{(miss>0).sum()} / {df.shape[1]}**")
    if not top_miss.empty:
        lines.append("")
        lines.append("Top missing columns (first 30):")
        lines.append(top_miss.to_frame("missing_count").to_markdown())
    lines.append("")

    # Duplicates
    lines.append("## 4) Duplicates")
    dup_all = int(df.duplicated().sum())
    lines.append(f"- Exact duplicate rows: **{dup_all}**")
    if "fips_code" in df.columns and date_col:
        dup_keys = int(df.duplicated(subset=["fips_code", date_col]).sum())
        lines.append(f"- Duplicate `(fips_code, {date_col})` rows: **{dup_keys}**")
    lines.append("")

    # Date coverage by county
    if "fips_code" in df.columns and date_col is not None:
        lines.append("## 5) Date Coverage by County")
        g = df.groupby("fips_code", as_index=False).agg(
            first_date=(date_col, "min"),
            last_date=(date_col, "max"),
            observed_days=(date_col, pd.Series.nunique),
        )
        try:
            g["expected_days"] = (pd.to_datetime(g["last_date"]) - pd.to_datetime(g["first_date"])).dt.days + 1
            g["missing_days_est"] = g["expected_days"] - g["observed_days"]
        except Exception as e:
            lines.append(f"> [warn] date coverage calc issue: {e}")
        lines.append(g.sort_values("missing_days_est", ascending=False).head(20).to_markdown(index=False))
        if not args.no_artifacts:
            _write_artifact_csv(g.sort_values("fips_code"), art_coverage)
        lines.append("")

    # Outage labels / diagnostics
    label_cols = [c for c in [
        "any_out","num_out_per_day","minutes_out","customers_out",
        "cust_minute_area","pct_out_max","pct_out_area",
        "customers_total","coverage","minutes_in_local_day",
        "snapshots_count","snapshot_minutes","customers_out_mean",
        "pct_out_area_unified","pct_out_area_covered","pct_out_max_unified",
        "train_mask"
    ] if c in df.columns]

    if label_cols:
        lines.append("## 6) Outage Labels / Diagnostics")
        if "any_out" in df.columns:
            vc = df["any_out"].value_counts(dropna=False).to_dict()
            frac1 = _safe_pct(vc.get(1,0), len(df))
            lines.append(f"- `any_out` distribution: {vc} (positives: **{frac1:.2f}%**)")
        num_labels = [c for c in label_cols if pd.api.types.is_numeric_dtype(df[c])]
        if num_labels:
            desc = df[num_labels].describe().T
            lines.append("")
            lines.append("Summary stats (labels/diagnostics):")
            lines.append(desc.to_markdown())
        if not args.no_artifacts and "any_out" in df.columns:
            _write_artifact_csv(
                df["any_out"].value_counts(dropna=False).rename("count").reset_index().rename(columns={"index":"any_out"}),
                art_labeldist
            )
        lines.append("")

    # Weather feature presence
    lines.append("## 7) Weather Feature Presence")
    expected_wx = ["PRCP","TMAX","TMIN","WSF2","AWND","WSF5"]
    present_map = {c: (c in df.columns) for c in expected_wx}
    lines.append(f"- Expected weather columns present: {present_map}")
    lines.append("")

    # High correlation pairs
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) >= 2:
        lines.append(f"## 8) High Correlations (|r| ≥ {args.corr_threshold})")
        corr = df[num_cols].corr(numeric_only=True)
        thr = float(args.corr_threshold)
        pairs = []
        for i, c1 in enumerate(num_cols):
            vals = corr.iloc[i, i+1:].dropna()
            hits = vals[vals.abs() >= thr]
            for c2, r in hits.items():
                pairs.append((c1, c2, float(r)))
        if pairs:
            corr_df = pd.DataFrame(pairs, columns=["feature_1","feature_2","corr"]).sort_values(
                "corr", ascending=False, key=lambda s: s.abs()
            )
            lines.append(corr_df.head(50).to_markdown(index=False))
            if not args.no_artifacts:
                _write_artifact_csv(corr_df, art_highcorr)
        else:
            lines.append("None")
        lines.append("")

    # Feature–target correlations (top |r|)
    typical_targets = [c for c in ["any_out","minutes_out","customers_out"] if c in df.columns]
    if typical_targets:
        lines.append("## 9) Feature–Target Correlations (top |r|)")
        top_corr_all = []
        for tgt in typical_targets:
            lines.append(f"### Target: `{tgt}`")
            sub = df.select_dtypes(include="number").dropna(subset=[tgt])
            if tgt not in sub.columns:
                continue
            corrs = sub.corr(numeric_only=True)[tgt].drop(tgt).dropna()
            corrs = corrs.reindex(corrs.abs().sort_values(ascending=False).index)
            top = corrs.head(20).to_frame("corr").reset_index().rename(columns={"index":"feature"})
            top_corr_all.append(top.assign(target=tgt))
            if not top.empty:
                lines.append(top.to_markdown(index=False))
            else:
                lines.append("_n/a_")
            lines.append("")
        if not args.no_artifacts and top_corr_all:
            _write_artifact_csv(pd.concat(top_corr_all, ignore_index=True), art_topcorr_y)
    else:
        lines.append("## 9) Feature–Target Correlations")
        lines.append("_No typical targets found (any_out/minutes_out/customers_out)._")
        lines.append("")

    # Constant & near-zero-variance
    nunique = df.nunique(dropna=False)
    const_cols = nunique[nunique <= 1]
    nzv_cols = nunique[nunique / max(len(df),1) < 0.01]
    lines.append("## 10) Constant & Near-Zero-Variance")
    lines.append(f"- Constant columns: **{const_cols.shape[0]}**")
    if const_cols.shape[0] > 0:
        lines.append(const_cols.to_frame("unique_count").head(50).to_markdown())
    lines.append(f"- Near-zero-variance (unique_frac < 1%): **{nzv_cols.shape[0]}**")
    if nzv_cols.shape[0] > 0:
        lines.append(nzv_cols.to_frame("unique_count").head(50).to_markdown())
    lines.append("")

    # Skewness & IQR outliers
    if len(num_cols) > 0:
        lines.append("## 11) Skewness & IQR Outliers (numeric only)")
        try:
            sk = df[num_cols].skew(numeric_only=True).sort_values(ascending=False)
            lines.append("Top skewness (first 30):")
            lines.append(sk.head(30).to_frame("skew").to_markdown())
        except Exception as e:
            lines.append(f"> [warn] skewness error: {e}")
        q1 = df[num_cols].quantile(0.25)
        q3 = df[num_cols].quantile(0.75)
        iqr = (q3 - q1).replace(0, np.nan)
        upper = q3 + 1.5 * iqr
        lower = q1 - 1.5 * iqr
        outlier_counts = ((df[num_cols] > upper) | (df[num_cols] < lower)).sum().sort_values(ascending=False)
        lines.append("")
        lines.append("Outlier counts by IQR rule (top 50):")
        lines.append(outlier_counts.head(50).to_frame("outlier_count").to_markdown())
        lines.append("")

    # Memory
    lines.append("## 12) Memory Usage")
    bycol = df.memory_usage(deep=True).drop(labels="Index", errors="ignore").sort_values(ascending=False)
    lines.append(f"- Total: **{mem_mb:.2f} MB**")
    lines.append(bycol.head(50).to_frame("bytes").to_markdown())
    lines.append("")

    # ======== EXTRA OUTAGE-AWARE METRICS & ARTIFACTS ========

    # 13) Per-county per-month outage rate
    if {"fips_code","any_out"}.issubset(df.columns) and date_col:
        lines.append("## 13) Per-County, Per-Month Outage Rate")
        tmp = df.copy()
        tmp["_month"] = tmp[date_col].dt.to_period("M").astype(str)
        by = tmp.groupby(["fips_code","_month"])["any_out"].agg(rate="mean", n="size").reset_index()
        lines.append(by.head(20).to_markdown(index=False))
        if not args.no_artifacts:
            _write_artifact_csv(by, art_bycnty_mo)
        lines.append("")

    # 14) Severity threshold sweep for pct_out_max
    if "pct_out_max" in df.columns:
        lines.append("## 14) Outage Severity Threshold Sweep (pct_out_max)")
        thr_grid = [0.001, 0.005, 0.01, 0.02]
        rows = [{"pct_thr": t, "prevalence": (df["pct_out_max"] >= t).mean(), "n": len(df)} for t in thr_grid]
        sweep = pd.DataFrame(rows)
        lines.append(sweep.to_markdown(index=False))
        if not args.no_artifacts:
            _write_artifact_csv(sweep, art_severity)
        lines.append("")

    # 15) Temporal alignment / leakage check (±2 lags)
    if date_col is not None:
        lines.append("## 15) Temporal Cross-Correlation (±2 days)")
        features_to_check = [c for c in ["PRCP","TMAX","WSF2","PRCP_7d_sum","WSF2_7d_max"] if c in df.columns]
        targets_to_check = [c for c in ["minutes_out","any_out"] if c in df.columns]
        xcorr_rows = []
        for feat in features_to_check:
            for tgt in targets_to_check:
                dsub = df[[feat, tgt]].dropna()
                if len(dsub) == 0:
                    continue
                xc = _xcorr_lags(dsub[feat], dsub[tgt], max_lag=2)
                xc["feature"] = feat
                xc["target"] = tgt
                xcorr_rows.append(xc)
        if xcorr_rows:
            xcorr_all = pd.concat(xcorr_rows, ignore_index=True)
            top = (xcorr_all.assign(abs_corr=xcorr_all["corr"].abs())
                   .sort_values("abs_corr", ascending=False)
                   .head(20)[["feature","target","lag","corr"]])
            lines.append(top.to_markdown(index=False))
            if not args.no_artifacts:
                _write_artifact_csv(xcorr_all[["feature","target","lag","corr"]], art_xcorr)
        else:
            lines.append("_n/a_")
        lines.append("")

    # 16) Integrity / consistency checks
    lines.append("## 16) Integrity / Consistency Checks")
    checks = {}
    if {"customers_out","customers_total"}.issubset(df.columns):
        checks["customers_out_leq_total_viol"] = int((df["customers_out"] > df["customers_total"]).sum())
    for col in ["pct_out_max","pct_out_area","pct_out_area_unified","pct_out_area_covered"]:
        if col in df.columns:
            checks[f"{col}_outside_0_1"] = int(((df[col] < 0) | (df[col] > 1)).sum())
    if {"minutes_out","snapshots_count","snapshot_minutes","coverage"}.issubset(df.columns):
        full = df["coverage"].ge(0.99)
        checks["minutes_out_mismatch_fullcov"] = int(
            (full & (df["minutes_out"] != df["snapshots_count"] * df["snapshot_minutes"])).sum()
        )
    lines.append(pd.Series(checks, dtype="object").to_frame("count").to_markdown())
    if not args.no_artifacts:
        _write_artifact_csv(pd.Series(checks, dtype="object").rename("count").reset_index().rename(columns={"index":"check"}), art_integrity)
    lines.append("")

    # 17) Redundant / identical numeric columns
    lines.append("## 17) Identical Numeric Columns")
    dupes = []
    num_cols = df.select_dtypes(include="number").columns
    seen = {}
    for c in num_cols:
        try:
            key = tuple(pd.util.hash_pandas_object(df[c].fillna(-9.99e15)).values)
        except Exception:
            continue
        if key in seen:
            dupes.append((seen[key], c))
        else:
            seen[key] = c
    if dupes:
        dupedf = pd.DataFrame(dupes, columns=["col_a","col_b"])
        lines.append(dupedf.head(50).to_markdown(index=False))
        if not args.no_artifacts:
            _write_artifact_csv(dupedf, art_identical)
    else:
        lines.append("None")
    lines.append("")

    # 18) Year-over-year drift (means of a few key cols)
    if date_col is not None:
        lines.append("## 18) Year-over-Year Means (selected)")
        year = df[date_col].dt.year
        cols_sel = [c for c in ["PRCP","TMAX","minutes_out","pct_out_max","customers_out"] if c in df.columns]
        yoy = df.assign(_year=year).groupby("_year")[cols_sel].mean(numeric_only=True).reset_index()
        if not yoy.empty:
            lines.append(yoy.to_markdown(index=False))
            if not args.no_artifacts:
                _write_artifact_csv(yoy, art_yearmeans)
        else:
            lines.append("_n/a_")
        lines.append("")

    # 19) Flag hierarchy checks (e.g., extreme_rain ⇒ heavy_rain)
    lines.append("## 19) Flag Hierarchy Checks")
    viol = {}
    if {"extreme_rain","heavy_rain"}.issubset(df.columns):
        viol["extreme_implies_heavy_viol"] = int(((df["extreme_rain"] == 1) & (df["heavy_rain"] == 0)).sum())
    if viol:
        lines.append(pd.Series(viol).to_frame("count").to_markdown())
        if not args.no_artifacts:
            _write_artifact_csv(pd.Series(viol).rename("count").reset_index().rename(columns={"index":"violation"}), art_flags)
    else:
        lines.append("None")
    lines.append("")

    # 20) Missingness by county & month (long table)
    if {"fips_code"}.issubset(df.columns) and date_col is not None:
        lines.append("## 20) Missingness by County × Month (long)")
        month = df[date_col].dt.to_period("M").astype(str)
        dmiss = []
        for c in df.columns:
            if c in ["fips_code", date_col]: 
                continue
            try:
                m = (df[c].isna().groupby([df["fips_code"], month]).mean()
                     .reset_index(name="missing_frac"))
                m["column"] = c
                dmiss.append(m)
            except Exception:
                continue
        if dmiss:
            miss_long = pd.concat(dmiss, ignore_index=True)
            lines.append(miss_long.head(20).to_markdown(index=False))
            if not args.no_artifacts:
                _write_artifact_csv(miss_long, art_miss_ctym)
        else:
            lines.append("_n/a_")
        lines.append("")

    # 21) Fold planning (year × county)
    if date_col is not None and "fips_code" in df.columns:
        lines.append("## 21) Fold Planning Summary (year × county)")
        d = df.copy()
        d["_year"] = d[date_col].dt.year
        agg = (d.groupby(["_year","fips_code"])
                 .agg(n=("day", "size") if "day" in d.columns else ("_year", "size"),
                      any_out_rate=("any_out","mean") if "any_out" in d.columns else ("_year", "size"))
                 .reset_index())
        lines.append(agg.head(30).to_markdown(index=False))
        if not args.no_artifacts:
            _write_artifact_csv(agg, art_folds)
        lines.append("")

    # Write report
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[report] wrote {report_path}")
    if not args.no_artifacts:
        print("[artifacts]")
        for pth in [art_coverage, art_highcorr, art_labeldist, art_bycnty_mo, art_severity,
                    art_xcorr, art_integrity, art_identical, art_yearmeans, art_flags,
                    art_topcorr_y, art_miss_ctym, art_folds]:
            print(" -", pth)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)
