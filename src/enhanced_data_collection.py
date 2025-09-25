# src/enhanced_data_collection.py
# Scaled weather data collection system for power outage prediction
# Design this to efficiently collect multi-year, multi-county weather data

import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import pandas as pd
import requests
from dotenv import load_dotenv

from config.target_counties import TARGET_COUNTIES, DATA_COLLECTION_CONFIG, CountyConfig


class EnhancedWeatherDataCollector:
    """
    I implement a robust, scalable weather data collection system
    This handles the complexity of collecting 5 years of data across multiple counties
    """

    def __init__(self, base_url: str = "https://www.ncei.noaa.gov/cdo-web/api/v2"):
        self.base_url = base_url
        self.session = requests.Session()  # I reuse connections for efficiency
        self.rate_limiter = threading.Semaphore(5)  # I enforce 5 req/sec limit
        self.last_request_time = 0
        self.setup_logging()
        self.data_dir_raw = Path("data/raw")
        self.data_dir_processed = Path("data/processed")
        self.data_dir_interim = Path("data/interim")  # I use this for temporary storage
        self._ensure_directories()

    def setup_logging(self):
        """
        Configure logging to track data collection progress and issues
        This is essential for debugging multi-county, multi-year collection
        """
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("data_collection.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def _ensure_directories(self):
        """Create necessary directory structure for organized data storage"""
        for directory in [
            self.data_dir_raw,
            self.data_dir_processed,
            self.data_dir_interim,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        # I create subdirectories for each year to organize large datasets
        for year in range(
            DATA_COLLECTION_CONFIG["start_year"], DATA_COLLECTION_CONFIG["end_year"] + 1
        ):
            (self.data_dir_raw / str(year)).mkdir(exist_ok=True)
            (self.data_dir_interim / str(year)).mkdir(exist_ok=True)

    def _auth_headers(self) -> Dict[str, str]:
        """Retrieve and validate API authentication token"""
        token = os.getenv("NOAA_CDO_TOKEN")
        if not token:
            raise RuntimeError("NOAA_CDO_TOKEN not set. Put it in config/.env")
        return {"token": token}

    def _rate_limited_request(self, url: str, params: Dict) -> requests.Response:
        """
        Implement rate limiting to respect NOAA's 5 requests/second limit
        This prevents API throttling during large-scale data collection
        """
        with self.rate_limiter:
            # I ensure minimum 500ms between requests (2 req/sec max) to be gentler on NOAA servers
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < 0.5:
                time.sleep(0.5 - time_since_last)

            response = self.session.get(
                url, headers=self._auth_headers(), params=params, timeout=60
            )
            self.last_request_time = time.time()
            response.raise_for_status()
            return response

    def _paged_get(self, endpoint: str, params: Dict) -> List[Dict]:
        """
        Implement efficient pagination for large data requests
        This handles NOAA's pagination system for multi-year data collection
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        all_rows: List[Dict] = []
        limit = 1000
        offset = 1

        while True:
            p = dict(params or {})
            p.update({"limit": limit, "offset": offset})

            try:
                resp = self._rate_limited_request(url, p)
                payload = resp.json()
                rows = payload.get("results", [])
                all_rows.extend(rows)
                # check if we've retrieved all available data
                meta = payload.get("metadata", {}).get("resultset", {})
                count = meta.get("count")

                if count is None:
                    if len(rows) < limit:
                        break
                else:
                    if offset + limit > count:
                        break
                offset += limit

            except requests.exceptions.RequestException as e:
                self.logger.error(
                    f"Request failed for {endpoint} with params {params}: {e}"
                )
                # I implement exponential backoff with longer delays for server errors
                if "503" in str(e) or "502" in str(e) or "504" in str(e):
                    # Server overload - wait longer and retry fewer times
                    retry_delay = min(30 + (offset // 1000) * 10, 120)
                    self.logger.warning(
                        f"Server error detected, waiting {retry_delay} seconds..."
                    )
                    time.sleep(retry_delay)
                else:
                    # Other errors - shorter backoff
                    time.sleep(min(2 ** (offset // 1000), 60))
                # limit retries to prevent infinite loops on persistent server errors
                if offset > 10000:  # After many retries, give up : (
                    self.logger.error("Too many retries, giving up on this request")
                    break
                continue
        return all_rows

    def collect_stations_for_county(
        self, county: CountyConfig, year: int
    ) -> pd.DataFrame:
        """
        Collect weather stations for a specific county and year
        This ensures we have consistent station coverage across the time period
        """
        self.logger.info(
            f"Collecting stations for {county.name}, {county.state} - {year}"
        )
        # set date range for the full year
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        params = {
            "datasetid": DATA_COLLECTION_CONFIG["dataset_id"],
            "locationid": county.fips_code,
            "startdate": start_date,
            "enddate": end_date,
        }

        try:
            stations_data = self._paged_get("stations", params)
            df = pd.DataFrame(stations_data)
            stations_file = (
                self.data_dir_raw
                / str(year)
                / f"stations_{county.fips_code.replace(':', '_')}_{year}.json"
            )
            with open(stations_file, "w") as f:
                json.dump(stations_data, f, indent=2)

            self.logger.info(f"Found {len(df)} stations for {county.name} in {year}")
            return df

        except Exception as e:
            self.logger.error(
                f"Failed to collect stations for {county.name} {year}: {e}"
            )
            return pd.DataFrame()

    def collect_weather_data_for_county_year(
        self, county: CountyConfig, year: int
    ) -> pd.DataFrame:
        """
        Collect comprehensive weather data for a county-year combination
        This is the core data collection function for building training datasets
        """
        self.logger.info(
            f"Collecting weather data for {county.name}, {county.state} - {year}"
        )
        # I collect data in quarterly chunks to manage API limits and memory
        quarterly_data = []
        for quarter in range(1, 5):
            quarter_df = self._collect_quarterly_data(county, year, quarter)
            if not quarter_df.empty:
                quarterly_data.append(quarter_df)
        if not quarterly_data:
            self.logger.warning(f"No weather data collected for {county.name} {year}")
            return pd.DataFrame()

        # combine quarterly data and apply basic processing
        annual_df = pd.concat(quarterly_data, ignore_index=True)
        processed_df = self._process_raw_weather_data(annual_df)
        self._save_county_year_data(county, year, annual_df, processed_df)

        return processed_df

    def _collect_quarterly_data(
        self, county: CountyConfig, year: int, quarter: int
    ) -> pd.DataFrame:
        """
        Collect weather data for a specific quarter to manage large requests
        Quarterly chunks prevent API timeouts and memory issues
        """
        # calculate quarter date ranges
        quarter_starts = {1: "01-01", 2: "04-01", 3: "07-01", 4: "10-01"}
        quarter_ends = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}
        start_date = f"{year}-{quarter_starts[quarter]}"
        end_date = f"{year}-{quarter_ends[quarter]}"
        # use all available datatypes for comprehensive feature engineering
        all_datatypes = (
            DATA_COLLECTION_CONFIG["base_datatypes"]
            + DATA_COLLECTION_CONFIG["extended_datatypes"]
        )
        params = {
            "datasetid": DATA_COLLECTION_CONFIG["dataset_id"],
            "locationid": county.fips_code,
            "startdate": start_date,
            "enddate": end_date,
            "units": DATA_COLLECTION_CONFIG["units"],
        }
        # I handle multiple datatypes by making separate requests if needed
        try:
            # first try with all datatypes
            params["datatypeid"] = all_datatypes
            data = self._paged_get("data", params)

            if data:
                df = pd.DataFrame(data)
                self.logger.info(
                    f"Collected {len(df)} records for {county.name} Q{quarter} {year}"
                )
                return df
            else:
                # fall back to base datatypes if extended ones aren't available
                params["datatypeid"] = DATA_COLLECTION_CONFIG["base_datatypes"]
                data = self._paged_get("data", params)
                df = pd.DataFrame(data)
                self.logger.info(
                    f"Collected {len(df)} base records for {county.name} Q{quarter} {year}"
                )
                return df

        except Exception as e:
            self.logger.error(
                f"Failed to collect Q{quarter} {year} data for {county.name}: {e}"
            )
            return pd.DataFrame()

    def _process_raw_weather_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw weather data into a clean format for feature engineering
        This standardizes the data structure across all counties and years
        """
        if df.empty:
            return df

        # normalize dates and create consistent day column
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df["day"] = df["date"].dt.date

        # I convert values to numeric, handling missing data appropriately
        if "value" in df.columns:
            df["value"] = pd.to_numeric(df["value"], errors="coerce")

        # I pivot to wide format for easier feature engineering
        try:
            wide_df = df.pivot_table(
                index=["day", "station"],
                columns="datatype",
                values="value",
                aggfunc="first",  # I use first value if duplicates exist
            )

            # I aggregate to county-day level using meteorologically appropriate methods
            county_daily = self._aggregate_to_county_daily(wide_df)
            return county_daily

        except Exception as e:
            self.logger.error(f"Failed to process weather data: {e}")
            return pd.DataFrame()

    def _aggregate_to_county_daily(self, station_df: pd.DataFrame) -> pd.DataFrame:
        """
        I aggregate station-level data to county-daily features
        Each weather variable requires different aggregation logic based on meteorology
        """
        aggregation_rules = {
            # Precipitation: sum across stations (total area precipitation)
            "PRCP": "sum",
            # Temperature: mean across stations (representative temperature)
            "TMAX": "mean",
            "TMIN": "mean",
            # Wind: max across stations (peak risk to infrastructure)
            "WSF2": "max",
            "WSF5": "max",
            "AWND": "mean",
            # Weather indicators: any occurrence across stations
            "WT01": "max",
            "WT02": "max",
            "WT03": "max",
            "WT04": "max",
            "WT05": "max",
            "WT06": "max",
            "WT08": "max",
            "WT11": "max",
        }
        # apply appropriate aggregation for each available column
        agg_dict = {}
        for col in station_df.columns:
            if col in aggregation_rules:
                agg_dict[col] = aggregation_rules[col]

        if not agg_dict:
            self.logger.warning("No recognized weather variables found for aggregation")
            return pd.DataFrame()
        daily_df = station_df.groupby("day").agg(agg_dict).reset_index()
        daily_df["day"] = pd.to_datetime(daily_df["day"])

        return daily_df

    def _save_county_year_data(
        self,
        county: CountyConfig,
        year: int,
        raw_df: pd.DataFrame,
        processed_df: pd.DataFrame,
    ):
        """
        Save data in organized file structure for easy access and processing
        This maintains data lineage and enables efficient partial reprocessing
        """
        county_id = county.fips_code.replace(":", "_")
        # save raw data for potential reprocessing
        raw_file = self.data_dir_raw / str(year) / f"weather_{county_id}_{year}.json"
        raw_df.to_json(raw_file, orient="records", indent=2, date_format="iso")

        # save processed data for immediate use
        processed_file = (
            self.data_dir_processed / f"weather_features_{county_id}_{year}.csv"
        )
        processed_df.to_csv(processed_file, index=False)
        self.logger.info(
            f"Saved data for {county.name} {year}: "
            f"{len(raw_df)} raw records, {len(processed_df)} daily features"
        )

    def collect_all_counties_parallel(
        self, max_workers: int = 3
    ) -> Dict[str, List[str]]:
        """
        Orchestrate parallel data collection across counties and years
        Limited parallelism respects API rate limits while improving efficiency
        """
        self.logger.info("Starting parallel data collection for all counties")
        collection_results = {"successful": [], "failed": []}

        # create tasks for each county-year combination
        tasks = []
        start_year = DATA_COLLECTION_CONFIG["start_year"]
        end_year = DATA_COLLECTION_CONFIG["end_year"]
        self.logger.info(f"Collection years from config: {start_year} to {end_year}")
        
        for county in TARGET_COUNTIES:
            for year in range(start_year, end_year + 1):
                tasks.append((county, year))

        # process tasks with limited parallelism to respect rate limits
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(
                    self.collect_weather_data_for_county_year, county, year
                ): (county, year)
                for county, year in tasks
            }
            for future in as_completed(future_to_task):
                county, year = future_to_task[future]
                try:
                    result_df = future.result()
                    if not result_df.empty:
                        collection_results["successful"].append(f"{county.name}_{year}")
                    else:
                        collection_results["failed"].append(f"{county.name}_{year}")
                except Exception as e:
                    self.logger.error(
                        f"Collection failed for {county.name} {year}: {e}"
                    )
                    collection_results["failed"].append(f"{county.name}_{year}")

        # log final collection summary
        self.logger.info(
            f"Collection complete: {len(collection_results['successful'])} successful, "
            f"{len(collection_results['failed'])} failed"
        )

        return collection_results

    def create_unified_dataset(self) -> pd.DataFrame:
        """
        I combine all collected county-year data into a unified training dataset
        This creates the final dataset for machine learning model development
        """
        self.logger.info("Creating unified dataset from all collected data")

        all_dataframes = []
        # iterate through all processed files
        for county in TARGET_COUNTIES:
            county_id = county.fips_code.replace(":", "_")
            county_data = []
            for year in range(
                DATA_COLLECTION_CONFIG["start_year"],
                DATA_COLLECTION_CONFIG["end_year"] + 1,
            ):
                file_path = (
                    self.data_dir_processed / f"weather_features_{county_id}_{year}.csv"
                )
                if file_path.exists():
                    year_df = pd.read_csv(file_path)
                    year_df["county_fips"] = county.fips_code
                    year_df["county_name"] = county.name
                    year_df["state"] = county.state
                    year_df["climate_zone"] = county.climate_zone
                    year_df["outage_risk_profile"] = county.outage_risk_profile
                    year_df["year"] = year
                    county_data.append(year_df)
                    
            if county_data:
                county_combined = pd.concat(county_data, ignore_index=True)
                all_dataframes.append(county_combined)
                self.logger.info(
                    f"Added {len(county_combined)} records for {county.name}"
                )

        if not all_dataframes:
            self.logger.error("No data files found for unified dataset creation")
            return pd.DataFrame()

        # I create the final unified dataset
        unified_df = pd.concat(all_dataframes, ignore_index=True)
        unified_df["day"] = pd.to_datetime(unified_df["day"])
        unified_df = unified_df.sort_values(["county_fips", "day"]).reset_index(
            drop=True
        )
        # save and log the unified dataset
        unified_file = self.data_dir_processed / "unified_weather_features_5year.csv"
        self.logger.info(
            f"Created unified dataset: {len(unified_df)} records across "
            f"{unified_df['county_fips'].nunique()} counties"
        )

        return unified_df


# def main():
#     load_dotenv("config/.env")

#     collector = EnhancedWeatherDataCollector()
#     test_county = TARGET_COUNTIES[0]  # Harris County
#     test_year = 2023

#     print(f"Testing data collection for {test_county.name}, {test_year}")
#     test_df = collector.collect_weather_data_for_county_year(test_county, test_year)
#     if not test_df.empty:
#         print(f"Test successful: {len(test_df)} daily records collected")
#         print(f"Available features: {list(test_df.columns)}")
#         # Proceed with full collection if test passes
#         print("\nStarting full data collection...")
#         results = collector.collect_all_counties_parallel(max_workers=2)

#         print(f"Collection results: {results}")

#         unified_df = collector.create_unified_dataset()
#         print(f"Final unified dataset: {len(unified_df)} records")

#     else:
#         print("Test failed - check API credentials and connectivity")


# if __name__ == "__main__":
#     main()
