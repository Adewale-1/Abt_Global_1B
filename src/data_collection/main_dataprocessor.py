# I created this main entry point as a main point of access  to run the complete data collection and processing workflow
# This orchestrates the entire process from raw weather data to ML-ready features following clean architecture

import os
import sys
import argparse
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.enhanced_data_collection import EnhancedWeatherDataCollector
from src.feature_engineering import PowerOutageFeatureEngineer
from config.target_counties import TARGET_COUNTIES, DATA_COLLECTION_CONFIG
from dotenv import load_dotenv


class MainDataProcessor:
    """
    Implement the main data processing orchestrator for power outage prediction
    This provides a single entry point for teammates to collect and process all weather data
    """

    def __init__(self, output_base_dir: str = "data"):
        self.base_dir = Path(output_base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        self.ml_ready_dir = self.base_dir / "ml_ready"
        self.data_collector = EnhancedWeatherDataCollector()
        self.feature_engineer = PowerOutageFeatureEngineer()

        self._setup_logging()

    def _setup_logging(self):
        """
        Configures logging to track the data processing workflow
        This helps teammates debug issues and monitor progress
        """
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("data_processing.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def validate_environment(self) -> bool:
        """
        Checks if the environment is properly configured before starting
        This prevents teammates from running into common setup issues
        """
        self.logger.info("Validating environment configuration...")
        # I check for required API token
        if not os.getenv("NOAA_CDO_TOKEN"):
            self.logger.error("NOAA_CDO_TOKEN environment variable not set")
            self.logger.error(
                "Please obtain a token from https://www.ncei.noaa.gov/cdo-web/token"
            )
            self.logger.error(
                "Then set it in .env file (in project root or config/ directory)"
            )
            return False
        # Ensures output directories exist
        for directory in [self.raw_dir, self.processed_dir, self.ml_ready_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        self.logger.info("Environment validation completed successfully")
        return True

    def collect_weather_data(
        self, start_year: Optional[int] = None, end_year: Optional[int] = None
    ) -> Dict[str, List[str]]:
        """
        I orchestrate the weather data collection process for all target counties
        This downloads raw weather data from NOAA for the specified time period
        """
        # I use configuration defaults if years not specified, but respect user overrides
        if start_year is None:
            start_year = DATA_COLLECTION_CONFIG["start_year"]
        if end_year is None:
            end_year = DATA_COLLECTION_CONFIG["end_year"]

        # log what years we're actually using
        self.logger.info(
            f"Arguments received - start_year: {start_year}, end_year: {end_year}"
        )
        self.logger.info(
            f"Starting weather data collection for {start_year}-{end_year}"
        )
        self.logger.info(f"Collecting data for {len(TARGET_COUNTIES)} counties")
        for county in TARGET_COUNTIES:
            self.logger.info(
                f"  - {county.name}, {county.state} ({county.outage_risk_profile})"
            )

        # I create a custom data collector with the specific year range to avoid race conditions
        results = self._collect_data_for_year_range(start_year, end_year)
        # Log the collection results
        successful_count = len(results["successful"])
        failed_count = len(results["failed"])
        self.logger.info(
            f"Data collection completed: {successful_count} successful, {failed_count} failed"
        )
        if failed_count > 0:
            self.logger.warning("Failed collections:")
            for failure in results["failed"]:
                self.logger.warning(f"  - {failure}")

        return results

    def _collect_data_for_year_range(
        self, start_year: int, end_year: int
    ) -> Dict[str, List[str]]:
        """
        I collect data sequentially one county at a time to be extremely gentle on NOAA servers
        This approach processes each county completely before moving to the next
        """
        import time

        collection_results = {"successful": [], "failed": []}

        # I calculate total work for progress tracking
        total_counties = len(TARGET_COUNTIES)
        total_years = end_year - start_year + 1
        total_tasks = total_counties * total_years

        self.logger.info(
            f"Sequential collection for {total_counties} counties, years {start_year}-{end_year}"
        )
        self.logger.info(f"Total tasks: {total_tasks} county-year combinations")

        task_count = 0
        # I process each county completely before moving to the next
        for county_idx, county in enumerate(TARGET_COUNTIES, 1):
            self.logger.info(
                f"Processing County {county_idx}/{total_counties}: {county.name}, {county.state}"
            )
            self.logger.info(f"   Risk Profile: {county.outage_risk_profile}")

            # I process all years for this county
            for year in range(start_year, end_year + 1):
                task_count += 1
                self.logger.info(
                    f"   Year {year} ({task_count}/{total_tasks}) - Starting collection..."
                )
                try:
                    # I collect data for this specific county-year combination
                    result_df = (
                        self.data_collector.collect_weather_data_for_county_year(
                            county, year
                        )
                    )
                    if not result_df.empty:
                        collection_results["successful"].append(f"{county.name}_{year}")
                        self.logger.info(
                            f"   Success: {len(result_df)} daily records collected"
                        )
                    else:
                        collection_results["failed"].append(f"{county.name}_{year}")
                        self.logger.warning(
                            f"   No data returned for {county.name} {year}"
                        )
                    # small delay between county-year combinations to be extra gentle
                    if task_count < total_tasks:  # Don't wait after the last task
                        self.logger.info(
                            "   Pausing 10 seconds before next collection..."
                        )
                        time.sleep(10)
                        
                except Exception as e:
                    collection_results["failed"].append(f"{county.name}_{year}")
                    self.logger.error(
                        f"   Collection failed for {county.name} {year}: {e}"
                    )
                    # longer pause after errors to let servers recover
                    if "503" in str(e) or "Server Error" in str(e):
                        self.logger.warning(
                            "   Server error detected - pausing 30 seconds..."
                        )
                        time.sleep(30)

            # longer pause between counties
            if county_idx < total_counties:
                self.logger.info(
                    f"   Completed {county.name} - pausing 20 seconds before next county..."
                )
                time.sleep(20)

        # final summary
        successful_count = len(collection_results["successful"])
        failed_count = len(collection_results["failed"])
        success_rate = (successful_count / total_tasks * 100) if total_tasks > 0 else 0
        self.logger.info(f"Sequential collection completed!")
        self.logger.info(
            f"   Success rate: {success_rate:.1f}% ({successful_count}/{total_tasks})"
        )

        return collection_results

    def create_unified_dataset(self) -> Optional[Path]:
        """
        I combine all collected county data into a single unified dataset
        This creates the foundation dataset for feature engineering
        """
        self.logger.info("Creating unified dataset from collected data...")

        try:
            unified_df = self.data_collector.create_unified_dataset()

            if unified_df.empty:
                self.logger.error("No data found to create unified dataset")
                return None
            # Save the unified dataset
            unified_path = self.processed_dir / "unified_weather_dataset.csv"
            unified_df.to_csv(unified_path, index=False)
            self.logger.info(f"Created unified dataset: {len(unified_df)} records")
            self.logger.info(f"Saved to: {unified_path}")

            return unified_path

        except Exception as e:
            self.logger.error(f"Failed to create unified dataset: {e}")
            return None

    def engineer_features(self, unified_dataset_path: Path) -> Optional[Path]:
        """
        Apply comprehensive feature engineering to the unified dataset
        This creates ML-ready features from the raw weather data
        """
        self.logger.info("Starting feature engineering process...")

        try:
            # load the unified dataset
            df = pd.read_csv(unified_dataset_path)
            self.logger.info(
                f"Loaded dataset with {len(df)} records and {len(df.columns)} columns"
            )
            # group by county to apply feature engineering separately
            engineered_dataframes = []
            for county_fips in df["county_fips"].unique():
                self.logger.info(f"Engineering features for {county_fips}")
                # Extract county data
                county_data = df[df["county_fips"] == county_fips].copy()
                county_data = county_data.sort_values("day").reset_index(drop=True)
                # separate metadata from weather data
                metadata_cols = [
                    "county_fips",
                    "county_name",
                    "state",
                    "climate_zone",
                    "outage_risk_profile",
                    "year",
                ]
                metadata_df = county_data[metadata_cols].copy()
                weather_cols = [
                    col for col in county_data.columns if col not in metadata_cols
                ]
                weather_data = county_data[weather_cols].copy()
                # Apply feature engineering
                engineered_weather = self.feature_engineer.engineer_all_features(
                    weather_data
                )
                # Combine engineered features with metadata
                engineered_weather_reset = engineered_weather.reset_index()
                engineered_county = pd.concat(
                    [
                        engineered_weather_reset.reset_index(drop=True),
                        metadata_df.reset_index(drop=True),
                    ],
                    axis=1,
                )
                engineered_dataframes.append(engineered_county)
                self.logger.info(
                    f"Created {len(engineered_weather.columns)} features for {county_fips}"
                )
            # combine all counties
            final_df = pd.concat(engineered_dataframes, ignore_index=True)
            final_df = final_df.sort_values(["county_fips", "day"]).reset_index(
                drop=True
            )
            # save the ML-ready dataset
            ml_ready_path = self.ml_ready_dir / "power_outage_ml_features.csv"
            final_df.to_csv(ml_ready_path, index=False)
            self.logger.info(
                f"Feature engineering completed: {len(final_df)} records with {len(final_df.columns)} features"
            )
            self.logger.info(f"Saved ML-ready dataset to: {ml_ready_path}")

            return ml_ready_path

        except Exception as e:
            self.logger.error(f"Feature engineering failed: {e}")
            return None

    def generate_summary_report(self, ml_dataset_path: Path):
        """
        Create a summary report of the processed data
        This helps me understand what data has been collected and processed
        """
        try:
            df = pd.read_csv(ml_dataset_path)
            report_lines = [
                "=" * 80,
                "POWER OUTAGE PREDICTION DATA PROCESSING SUMMARY",
                "=" * 80,
                f"Processing completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "DATASET OVERVIEW:",
                f"  Total records: {len(df):,}",
                f"  Total features: {len(df.columns)}",
                f"  Date range: {df['day'].min()} to {df['day'].max()}",
                f"  Counties covered: {df['county_fips'].nunique()}",
                "",
                "COUNTIES PROCESSED:",
            ]
            for county_info in (
                df.groupby(["county_name", "state"])["day"].count().items()
            ):
                county_name, state = county_info[0]
                record_count = county_info[1]
                report_lines.append(
                    f"  - {county_name}, {state}: {record_count} daily records"
                )
            report_lines.extend(
                [
                    "",
                    "FEATURE CATEGORIES:",
                    "  - Temperature stress features (thermal cycling, degree days, extremes)",
                    "  - Precipitation impact features (accumulation, intensity, patterns)",
                    "  - Wind damage features (speed categories, persistency, acceleration)",
                    "  - Compound risk features (ice storms, multiple hazards)",
                    "  - Temporal patterns (seasonal effects, consecutive events)",
                    "",
                    "EXTREME WEATHER EVENTS DETECTED:",
                ]
            )
            # Summarize extreme weather events
            extreme_cols = [
                col
                for col in df.columns
                if any(
                    x in col.lower()
                    for x in ["extreme", "ice_storm", "damaging", "heavy"]
                )
            ]
            for col in extreme_cols[:10]:  # Show first 10 extreme features
                event_count = (
                    df[col].sum() if df[col].dtype in ["int64", "float64"] else 0
                )
                if event_count > 0:
                    report_lines.append(f"  - {col}: {event_count} events total")

            # save and display the report
            report_text = "\n".join(report_lines)
            report_path = self.ml_ready_dir / "data_processing_summary.txt"
            with open(report_path, "w") as f:
                f.write(report_text)
            print(report_text)
            self.logger.info(f"Summary report saved to: {report_path}")

        except Exception as e:
            self.logger.error(f"Failed to generate summary report: {e}")

    def run_complete_workflow(
        self, start_year: Optional[int] = None, end_year: Optional[int] = None
    ) -> bool:
        """
        Execute the complete data processing workflow from start to finish
        This is the main method teammates should call to process all data
        """
        self.logger.info(
            f"Starting complete data processing workflow with years: {start_year} to {end_year}"
        )
        # validate the environment first
        if not self.validate_environment():
            return False

        try:
            # collect weather data
            collection_results = self.collect_weather_data(start_year, end_year)
            if len(collection_results["successful"]) == 0:
                self.logger.error("No data was successfully collected")
                return False
            # create unified dataset
            unified_path = self.create_unified_dataset()
            if not unified_path:
                return False
            # engineer features
            ml_ready_path = self.engineer_features(unified_path)
            if not ml_ready_path:
                return False
            # generate summary report
            self.generate_summary_report(ml_ready_path)
            self.logger.info("Complete workflow finished successfully!")
            return True
        except Exception as e:
            self.logger.error(f"Workflow failed: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Power Outage Prediction Data Processor - Collect and process weather data for ML model training"
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=DATA_COLLECTION_CONFIG["start_year"],
        help=f"Start year for data collection (default: {DATA_COLLECTION_CONFIG['start_year']})",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=DATA_COLLECTION_CONFIG["end_year"],
        help=f"End year for data collection (default: {DATA_COLLECTION_CONFIG['end_year']})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for processed data (default: data)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate environment setup without processing data",
    )
    args = parser.parse_args()
    # I try to load from both possible locations, I personally prefer the project root .env file but my teammates could put in the config folder which makes sense.
    env_loaded = load_dotenv(project_root / ".env") or load_dotenv(
        project_root / "config" / ".env"
    )
    processor = MainDataProcessor(output_base_dir=args.output_dir)

    if args.validate_only:
        success = processor.validate_environment()
        if success:
            print("Environment validation successful! Ready to process data.")
        else:
            print(
                "Environment validation failed. Please check the error messages above."
            )
        return
    # run the complete workflow
    success = processor.run_complete_workflow(args.start_year, args.end_year)
    
    if success:
        print("\nData processing completed successfully!")
        print(
            f"Check the {args.output_dir}/ml_ready/ directory"
        )
    else:
        print("\nData processing failed. Please check the log messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
