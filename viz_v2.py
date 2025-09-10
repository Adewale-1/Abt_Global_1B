# viz_v2.py  (Python 3.11+)
# Visualize NOAA CDO v2 outputs produced by cdo_v2_demo.py
# Defaults:
#   processed CSV: data/processed/cdo_v2_harris_tx_uri_window.csv
#   raw JSON:      data/raw/cdo_v2_sample_data.json
# Outputs PNGs into results/figures/
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def load_processed(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "day" not in df.columns:
        raise ValueError("Expected a 'day' column in processed CSV.")
    df["day"] = pd.to_datetime(df["day"], utc=True)
    df = df.set_index("day").sort_index()
    for col in ["PRCP","TMAX","TMIN","WSF2"]:
        if col not in df.columns:
            df[col] = float("nan")
    return df

def load_raw(raw_json_path: Path) -> pd.DataFrame | None:
    if not raw_json_path.exists():
        return None
    rdf = pd.read_json(raw_json_path)
    # Expected columns from cdo_v2_demo.py: date, station, datatype, value
    for c in ["date","station","datatype","value"]:
        if c not in rdf.columns:
            return None
    rdf["date"] = pd.to_datetime(rdf["date"], utc=True)
    rdf["day"] = rdf["date"].dt.floor("D")
    return rdf

def plot_overview(ts: pd.DataFrame, outpng: Path):
    fig, ax1 = plt.subplots(figsize=(10,5))
    ax1.bar(ts.index, ts["PRCP"], width=0.9, label="PRCP (mm)")
    ax1.set_ylabel("PRCP (mm)")
    ax1.set_xlabel("Date")

    ax2 = ax1.twinx()
    ax2.plot(ts.index, ts["TMAX"], marker="o", label="TMAX (°C)")
    ax2.plot(ts.index, ts["TMIN"], marker="o", label="TMIN (°C)")
    ax2.plot(ts.index, ts["WSF2"], marker="o", label="WSF2 (m/s)")

    ax2.set_ylabel("Temp (°C) / Wind (m/s)")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper left")
    fig.tight_layout()
    fig.savefig(outpng, dpi=150)
    plt.close(fig)

def plot_rolling(ts: pd.DataFrame, outpng: Path):
    roll = pd.DataFrame({
        "PRCP_7d_sum": ts["PRCP"].rolling(7, min_periods=1).sum(),
        "WSF2_7d_max": ts["WSF2"].rolling(7, min_periods=1).max(),
    }, index=ts.index)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(roll.index, roll["PRCP_7d_sum"], label="PRCP 7d sum (mm)")
    ax.plot(roll.index, roll["WSF2_7d_max"], label="WSF2 7d max (m/s)")
    ax.set_title("7-day Rolling Precip & Max Wind")
    ax.set_xlabel("Date"); ax.set_ylabel("Rolling values")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpng, dpi=150)
    plt.close(fig)

def plot_station_coverage(rdf: pd.DataFrame, outpng: Path):
    # daily count of unique stations that reported anything that day
    cover = (rdf.groupby("day")["station"].nunique()).rename("stations_reporting")
    fig, ax = plt.subplots(figsize=(10,3.5))
    ax.plot(cover.index, cover.values, marker="o")
    ax.set_title("CDO v2 Station Coverage (unique stations per day)")
    ax.set_xlabel("Date"); ax.set_ylabel("# Stations")
    fig.tight_layout()
    fig.savefig(outpng, dpi=150)
    plt.close(fig)

def plot_prcp_mean_vs_sum(rdf: pd.DataFrame, processed_ts: pd.DataFrame, outpng: Path):
    # compute mean PRCP across stations per day from raw JSON
    sub = rdf[rdf["datatype"] == "PRCP"][["day","station","value"]].dropna()
    mean_per_day = sub.groupby("day")["value"].mean().rename("PRCP_mean_from_raw")
    comp = processed_ts[["PRCP"]].rename(columns={"PRCP":"PRCP_sum_from_processed"}).join(mean_per_day, how="left")
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(comp.index, comp["PRCP_sum_from_processed"], marker="o", label="PRCP sum (processed CSV)")
    ax.plot(comp.index, comp["PRCP_mean_from_raw"], marker="o", label="PRCP mean (from raw JSON)")
    ax.set_title("PRCP aggregation choice: sum vs mean across stations")
    ax.set_xlabel("Date"); ax.set_ylabel("mm")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpng, dpi=150)
    plt.close(fig)

def main():
    p = argparse.ArgumentParser(description="Visualize NOAA CDO v2 outputs.")
    p.add_argument("--processed", default="data/processed/cdo_v2_harris_tx_uri_window.csv")
    p.add_argument("--raw", default="data/raw/cdo_v2_sample_data.json")
    p.add_argument("--outdir", default="results/figures")
    args = p.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    ts = load_processed(Path(args.processed))
    raw_df = load_raw(Path(args.raw))

    # 1) Overview plot (bars/lines)
    plot_overview(ts, outdir / "v2_overview.png")

    # 2) Rolling features
    plot_rolling(ts, outdir / "v2_rolling.png")

    # 3) Station coverage (if raw JSON available)
    if raw_df is not None:
        plot_station_coverage(raw_df, outdir / "v2_station_coverage.png")
        # 4) PRCP mean vs sum comparison (optional but useful)
        plot_prcp_mean_vs_sum(raw_df, ts, outdir / "v2_prcp_mean_vs_sum.png")

    print("Saved figures to:", outdir.resolve())

if __name__ == "__main__":
    main()
