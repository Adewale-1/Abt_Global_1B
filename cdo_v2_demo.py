# cdo_v2_demo.py
# Python 3.11+
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv


BASE = "https://www.ncei.noaa.gov/cdo-web/api/v2"
DATA_DIR_RAW = Path("data/raw")
DATA_DIR_PROCESSED = Path("data/processed")


def _auth_headers() -> Dict[str, str]:
    token = os.getenv("NOAA_CDO_TOKEN")
    if not token:
        raise RuntimeError("NOAA_CDO_TOKEN not set. Put it in config/.env (see .env.example).")
    return {"token": token}


def _paged_get(endpoint: str, params: Dict) -> List[Dict]:
    """
    Generic paginator for CDO v2 endpoints that return JSON with 'results' and 'metadata.resultset'.
    Automatically respects 5 req/s with a small sleep.
    """
    url = f"{BASE}/{endpoint.lstrip('/')}"
    headers = _auth_headers()

    all_rows: List[Dict] = []
    limit = 1000
    offset = 1

    while True:
        p = dict(params or {})
        p.update({"limit": limit, "offset": offset})
        resp = requests.get(url, headers=headers, params=p, timeout=60)
        resp.raise_for_status()
        payload = resp.json()

        rows = payload.get("results", [])
        all_rows.extend(rows)

        meta = payload.get("metadata", {}).get("resultset", {})
        count = meta.get("count")
        # If 'count' not present, fall back to "less than a page" heuristic.
        if count is None:
            if len(rows) < limit:
                break
            offset += limit
            time.sleep(0.25)
            continue

        if offset + limit > count:
            break

        offset += limit
        time.sleep(0.25)  # be nice to the API (≤5 req/s)

    return all_rows


def list_datasets(datasetid: Optional[str] = None) -> pd.DataFrame:
    params = {}
    if datasetid:
        params["datasetid"] = datasetid
    rows = _paged_get("datasets", params)
    return pd.DataFrame(rows)


def list_datatypes(datasetid: Optional[str] = None, datatypeid: Optional[str] = None) -> pd.DataFrame:
    params = {}
    if datasetid:
        params["datasetid"] = datasetid
    if datatypeid:
        params["datatypeid"] = datatypeid
    rows = _paged_get("datatypes", params)
    return pd.DataFrame(rows)


def list_stations(datasetid: str, locationid: Optional[str] = None,
                  startdate: Optional[str] = None, enddate: Optional[str] = None) -> pd.DataFrame:
    params = {"datasetid": datasetid}
    if locationid:
        params["locationid"] = locationid
    if startdate:
        params["startdate"] = startdate
    if enddate:
        params["enddate"] = enddate
    rows = _paged_get("stations", params)
    return pd.DataFrame(rows)


def fetch_data(datasetid: str,
               datatypes: List[str],
               locationid: Optional[str] = None,
               stationid: Optional[str] = None,
               startdate: str = None,
               enddate: str = None,
               units: str = "metric") -> pd.DataFrame:
    """
    Pulls observation records from /data endpoint.
    Returns tidy DataFrame with columns: date, station, datatype, value, attributes...
    """
    assert (locationid or stationid), "Provide either locationid (e.g., FIPS:48201) or stationid (e.g., GHCND:USW00012960)"
    params = {
        "datasetid": datasetid,
        "units": units,
        "startdate": startdate,
        "enddate": enddate,
    }
    # multiple datatypeid params allowed
    for dt in datatypes:
        params.setdefault("datatypeid", [])
        # requests handles repeated params if we pass list under same key
        params["datatypeid"].append(dt)

    if locationid:
        params["locationid"] = locationid
    if stationid:
        params["stationid"] = stationid

    rows = _paged_get("data", params)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # normalize date to UTC date (keep calendar day for daily summaries)
    # CDO daily 'date' will look like '2021-02-10T00:00:00'
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["day"] = df["date"].dt.date

    # make numeric if possible
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

    return df


def pivot_and_aggregate_to_county_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot station/day datatypes to columns, then aggregate to county-day features:
      - PRCP: sum across stations (mm)
      - TMAX/TMIN: mean across stations (°C)
      - WSF2: max across stations (m/s or dataset units)
    Assumes df has columns: day, station, datatype, value
    """
    if df.empty:
        return df

    # pivot to wide per (day, station)
    wide = df.pivot_table(
        index=["day", "station"],
        columns="datatype",
        values="value",
        aggfunc="max"  # if duplicates, keep max
    )

    # choose aggregations by variable
    aggregations = {}
    if "PRCP" in wide.columns:
        aggregations["PRCP"] = "sum"
    if "TMAX" in wide.columns:
        aggregations["TMAX"] = "mean"
    if "TMIN" in wide.columns:
        aggregations["TMIN"] = "mean"
    if "WSF2" in wide.columns:
        aggregations["WSF2"] = "max"

    agg = wide.groupby("day").agg(aggregations).reset_index()
    # reorder columns nicely
    cols = ["day"] + [c for c in ["PRCP", "TMAX", "TMIN", "WSF2"] if c in agg.columns]
    return agg[cols]


def ensure_dirs():
    DATA_DIR_RAW.mkdir(parents=True, exist_ok=True)
    DATA_DIR_PROCESSED.mkdir(parents=True, exist_ok=True)


def main():
    load_dotenv("config/.env")
    ensure_dirs()

    # ---- CONFIG: pick a county & dates (Winter Storm Uri window in TX) ----
    datasetid = "GHCND"                # Daily Summaries
    sample_fips = "FIPS:48201"         # Harris County, TX
    startdate = "2021-02-10"
    enddate   = "2021-02-20"
    datatypes = ["PRCP", "TMAX", "TMIN", "WSF2"]  # precip, temps, wind (fastest 2-min)

    print("1) List a few datasets (discovery)")
    ds = list_datasets()
    print(ds[["id", "name"]].head(10).to_string(index=False))

    print("\n2) List datatypes available for GHCND (preview)")
    dtypes = list_datatypes(datasetid=datasetid)
    print(dtypes[["id", "name"]].head(10).to_string(index=False))

    print(f"\n3) List stations for {datasetid} in {sample_fips} during {startdate}..{enddate}")
    stations = list_stations(datasetid=datasetid, locationid=sample_fips,
                             startdate=startdate, enddate=enddate)
    stations_path = DATA_DIR_RAW / "cdo_v2_sample_stations.json"
    stations.to_json(stations_path, orient="records", indent=2)
    print(f"Saved stations → {stations_path}  (n={len(stations)})")

    print(f"\n4) Fetch data for {datasetid} {datatypes} in {sample_fips} {startdate}..{enddate}")
    raw = fetch_data(datasetid=datasetid, datatypes=datatypes,
                     locationid=sample_fips, startdate=startdate, enddate=enddate,
                     units="metric")
    raw_path = DATA_DIR_RAW / "cdo_v2_sample_data.json"
    raw.to_json(raw_path, orient="records", indent=2, date_format="iso")
    print(f"Saved raw records → {raw_path}  (n={len(raw)})")
    if raw.empty:
        print("No records returned. Try a different county/date window.")
        return

    print("\n5) Pivot/aggregate to county-day features")
    daily = pivot_and_aggregate_to_county_day(raw)
    daily_path = DATA_DIR_PROCESSED / "cdo_v2_harris_tx_uri_window.csv"
    daily.to_csv(daily_path, index=False)
    print(f"Saved county-day features → {daily_path}")
    print("\nPreview:")
    print(daily.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
