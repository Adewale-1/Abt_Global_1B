# Feature engineering for power outage prediction


import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore")


@dataclass
class FeatureMetadata:
    """
    I use this structure to document each feature's purpose and importance
    This maintains feature documentation and helps with model interpretability
    """

    name: str
    description: str
    rationale: str
    outage_relevance: str
    data_type: str


class PowerOutageFeatureEngineer:
    """
    Implement comprehensive feature engineering specifically for power outage prediction
    Each feature group targets different failure modes of electrical infrastructure
    """

    def __init__(self):
        # feature metadata for documentation and model interpretation
        self.feature_metadata = self._initialize_feature_metadata()

    def engineer_all_features(self, weather_df: pd.DataFrame) -> pd.DataFrame:
        """
        Orchestrate the complete feature engineering pipeline
        This creates a comprehensive feature set targeting all major outage causes
        """
        # ensure the dataframe has the required structure
        df = self._validate_and_prepare_data(weather_df.copy())
        # apply feature engineering in logical groups
        df = self._create_lag_features(df)
        df = self._create_rolling_features(df)
        df = self._create_extreme_event_features(df)
        df = self._create_temperature_features(df)
        df = self._create_precipitation_features(df)
        df = self._create_wind_features(df)
        df = self._create_seasonal_features(df)
        df = self._create_compound_risk_features(df)
        df = self._create_infrastructure_stress_indicators(df)

        return df

    def _validate_and_prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This validates input data and ensures proper datetime indexing
        This prevents downstream errors and ensures consistent data structure
        """
        if "day" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df["day"] = pd.to_datetime(df["day"])
            df = df.set_index("day").sort_index()

        # I ensure all required base features exist
        required_features = ["PRCP", "TMAX", "TMIN", "WSF2"]
        for feature in required_features:
            if feature not in df.columns:
                df[feature] = np.nan

        return df

    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create lagged weather features because power outages often result from
        cumulative stress or delayed infrastructure failures from previous days
        """
        lag_configs = [
            (
                "PRCP",
                [1, 2, 3],
                "Cumulative moisture affects soil conditions and tree stability",
            ),
            (
                "TMAX",
                [1, 2, 3],
                "Temperature trends affect infrastructure thermal stress",
            ),
            ("TMIN", [1, 2, 3], "Cold snaps have delayed effects on infrastructure"),
            ("WSF2", [1, 2], "Wind damage may not be immediately apparent"),
        ]

        for feature, lags, rationale in lag_configs:
            for lag in lags:
                new_feature = f"{feature}_lag{lag}d"
                df[new_feature] = df[feature].shift(lag)

        return df

    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        I implement rolling window features to capture sustained weather patterns
        These identify prolonged conditions that gradually weaken infrastructure
        """
        rolling_configs = [
            # Precipitation accumulation features
            (
                "PRCP",
                "sum",
                [3, 7, 14],
                "Cumulative precipitation saturates soil and increases tree fall risk",
            ),
            (
                "PRCP",
                "max",
                [3, 7],
                "Peak precipitation intensity within windows shows flood risk",
            ),
            # Temperature stress features
            (
                "TMAX",
                "mean",
                [3, 7],
                "Sustained heat increases transformer failure rates",
            ),
            ("TMAX", "max", [7, 14], "Peak heat loads stress electrical equipment"),
            ("TMIN", "mean", [3, 7], "Sustained cold affects infrastructure materials"),
            (
                "TMIN",
                "min",
                [7, 14],
                "Extreme cold causes material contraction and failures",
            ),
            # Wind damage accumulation
            (
                "WSF2",
                "max",
                [3, 7],
                "Peak winds in window show maximum infrastructure stress",
            ),
            (
                "WSF2",
                "mean",
                [7],
                "Sustained winds cause fatigue in power line hardware",
            ),
        ]

        for feature, agg_func, windows, rationale in rolling_configs:
            for window in windows:
                new_feature = f"{feature}_{window}d_{agg_func}"
                if agg_func == "sum":
                    df[new_feature] = df[feature].rolling(window, min_periods=1).sum()
                elif agg_func == "mean":
                    df[new_feature] = df[feature].rolling(window, min_periods=1).mean()
                elif agg_func == "max":
                    df[new_feature] = df[feature].rolling(window, min_periods=1).max()
                elif agg_func == "min":
                    df[new_feature] = df[feature].rolling(window, min_periods=1).min()

        return df

    def _create_extreme_event_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary indicators for extreme weather events
        These capture threshold effects where outage risk increases dramatically
        """
        # Heavy precipitation thresholds (based on meteorological definitions)
        df["heavy_rain"] = (df["PRCP"] >= 25.4).astype(int)  # >1 inch in 24h
        df["extreme_rain"] = (df["PRCP"] >= 76.2).astype(int)  # >3 inches in 24h
        # Temperature extremes (varies by season, using general thresholds)
        df["heat_wave"] = (df["TMAX"] >= 35.0).astype(int)  # >95°F
        df["extreme_heat"] = (df["TMAX"] >= 40.0).astype(int)  # >104°F
        df["freezing"] = (df["TMIN"] <= 0.0).astype(int)  # At or below freezing
        df["extreme_cold"] = (df["TMIN"] <= -12.0).astype(int)  # <10°F
        # High wind events
        df["high_winds"] = (df["WSF2"] >= 17.9).astype(int)  # >40 mph
        df["damaging_winds"] = (df["WSF2"] >= 25.7).astype(
            int
        )  # >58 mph (severe threshold)
        # create consecutive day counters for sustained events
        df["consecutive_heat_days"] = self._count_consecutive_days(df["heat_wave"])
        df["consecutive_freeze_days"] = self._count_consecutive_days(df["freezing"])
        df["consecutive_rain_days"] = self._count_consecutive_days(df["PRCP"] > 0)

        return df

    def _create_temperature_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer temperature-specific features that affect power infrastructure
        Temperature variations cause thermal stress and change electrical demand patterns
        """
        # Diurnal temperature range - affects infrastructure thermal cycling
        df["temp_range_daily"] = df["TMAX"] - df["TMIN"]
        # Temperature volatility - rapid changes stress infrastructure
        df["temp_volatility_3d"] = df["TMAX"].rolling(3).std()
        df["temp_change_1d"] = df["TMAX"].diff(1).abs()
        # Degree day calculations for infrastructure stress
        df["heating_degree_days"] = np.maximum(18.3 - df["TMAX"], 0)  # Base 65°F
        df["cooling_degree_days"] = np.maximum(df["TMAX"] - 18.3, 0)
        # Freeze-thaw cycles - cause infrastructure damage
        df["freeze_thaw_cycle"] = ((df["TMIN"] <= 0) & (df["TMAX"] > 0)).astype(int)

        return df

    def _create_precipitation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Develop precipitation features that correlate with different outage mechanisms
        Different precipitation patterns cause distinct types of power system failures
        """
        # Precipitation intensity categories
        df["light_rain"] = ((df["PRCP"] > 0) & (df["PRCP"] < 2.5)).astype(int)
        df["moderate_rain"] = ((df["PRCP"] >= 2.5) & (df["PRCP"] < 25.4)).astype(int)
        # Wet/dry periods - affect soil moisture and tree stability
        df["days_since_rain"] = self._days_since_event(df["PRCP"] > 0)
        # Use rolling sum if available, otherwise use current day precipitation as indicator
        if "PRCP_7d_sum" in df.columns:
            df["wet_period_indicator"] = (df["PRCP_7d_sum"] > 50.0).astype(int)
        else:
            df["wet_period_indicator"] = (df["PRCP"] > 25.0).astype(
                int
            )  # Single day heavy rain as fallback

        # Precipitation variability
        df["precip_volatility_7d"] = df["PRCP"].rolling(7).std()

        return df

    def _create_wind_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create wind-related features as wind is the primary cause of power line failures
        Different wind patterns cause different types of infrastructure damage
        """
        # Wind speed categories (Beaufort scale adapted for power systems)
        df["moderate_winds"] = ((df["WSF2"] >= 8.9) & (df["WSF2"] < 17.9)).astype(
            int
        )  # 20-40 mph
        df["strong_winds"] = ((df["WSF2"] >= 17.9) & (df["WSF2"] < 25.7)).astype(
            int
        )  # 40-58 mph

        # Wind persistency - sustained winds cause fatigue failures
        # I check if the rolling feature exists first, if not create a simple version
        if "WSF2_3d_mean" in df.columns:
            df["sustained_winds_3d"] = (df["WSF2_3d_mean"] > 13.4).astype(int)
        else:
            # create a fallback using the base WSF2 feature
            df["sustained_winds_3d"] = (df["WSF2"] > 13.4).astype(int)

        # Wind acceleration - rapid wind increases are particularly damaging
        df["wind_acceleration_1d"] = df["WSF2"].diff(1)
        df["rapid_wind_increase"] = (df["wind_acceleration_1d"] > 8.9).astype(
            int
        )  # >20 mph increase

        return df

    def _create_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add seasonal features as outage risk varies significantly by season
        Different seasons have distinct weather patterns and infrastructure vulnerabilities
        """
        df["month"] = df.index.month
        df["day_of_year"] = df.index.dayofyear
        df["season"] = df["month"].map(
            {
                12: "winter",
                1: "winter",
                2: "winter",
                3: "spring",
                4: "spring",
                5: "spring",
                6: "summer",
                7: "summer",
                8: "summer",
                9: "fall",
                10: "fall",
                11: "fall",
            }
        )
        # create seasonal interaction terms for risk factors
        df["winter_freeze"] = ((df["season"] == "winter") & df["freezing"]).astype(int)
        df["summer_heat"] = ((df["season"] == "summer") & df["heat_wave"]).astype(int)
        df["spring_storms"] = ((df["season"] == "spring") & df["high_winds"]).astype(
            int
        )

        return df

    def _create_compound_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer compound risk features as multiple simultaneous weather factors
        dramatically increase outage probability beyond individual component risks
        """
        # Ice storm conditions - the most dangerous combination for power systems
        df["ice_storm_risk"] = (
            (df["TMIN"] <= 0) & (df["TMAX"] > 0) & (df["PRCP"] > 0)
        ).astype(int)
        # Wet and windy conditions - high tree fall risk
        df["wet_windy_combo"] = ((df["PRCP"] > 0) & (df["WSF2"] > 13.4)).astype(int)
        # Heat and high demand potential
        df["heat_demand_stress"] = (
            (df["TMAX"] > 32.0) & (df["cooling_degree_days"] > 10)
        ).astype(int)
        # Multiple extreme conditions
        df["multiple_extremes"] = (
            df["extreme_rain"]
            + df["extreme_heat"]
            + df["extreme_cold"]
            + df["damaging_winds"]
        ).clip(0, 1)

        return df

    def _create_infrastructure_stress_indicators(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        I develop infrastructure stress indicators that predict equipment failure likelihood
        These features model the cumulative stress on power system components
        """
        # Cumulative thermal stress index
        df["thermal_stress_index"] = (
            df["temp_range_daily"] * 0.3
            + df["temp_volatility_3d"] * 0.4
            + df["heating_degree_days"] * 0.1
            + df["cooling_degree_days"] * 0.2
        )
        # Mechanical stress from weather
        # I handle missing rolling features gracefully
        wsf2_component = (
            df["WSF2_7d_max"] * 0.5 if "WSF2_7d_max" in df.columns else df["WSF2"] * 0.5
        )
        wind_accel_component = (
            df["wind_acceleration_1d"].abs() * 0.3
            if "wind_acceleration_1d" in df.columns
            else 0
        )
        rain_component = (
            df["consecutive_rain_days"] * 0.2
            if "consecutive_rain_days" in df.columns
            else 0
        )
        df["mechanical_stress_index"] = (
            wsf2_component + wind_accel_component + rain_component
        )
        # Overall weather severity score
        df["weather_severity_score"] = (
            df["extreme_rain"] * 3
            + df["damaging_winds"] * 4
            + df["extreme_cold"] * 2
            + df["extreme_heat"] * 2
            + df["ice_storm_risk"] * 5
        )

        return df

    def _count_consecutive_days(self, series: pd.Series) -> pd.Series:
        """
        Implement this helper to count consecutive occurrences of events
        Consecutive extreme weather days have compounding effects on infrastructure
        """
        # I use a cumulative approach to count consecutive events
        groups = (series != series.shift()).cumsum()
        consecutive = series.groupby(groups).cumsum()
        return consecutive * series  # Reset to 0 when series is 0

    def _days_since_event(self, event_series: pd.Series) -> pd.Series:
        """
        Calculate days since the last occurrence of an event
        This captures the temporal relationship between weather events and infrastructure state
        """
        last_event = event_series.where(event_series).ffill()
        days_since = event_series.index.to_series() - last_event.index.to_series()[
            last_event.notna()
        ].reindex(event_series.index, method="ffill")
        return days_since.dt.days.fillna(999)  # I use 999 for never occurred

    def _initialize_feature_metadata(self) -> Dict[str, FeatureMetadata]:
        """
        Maintain comprehensive metadata for all engineered features
        This supports model interpretability and feature selection decisions
        """
        metadata = {}

        # I would populate this with detailed metadata for each feature
        # For brevity, I'm showing the pattern here
        base_features = {
            "PRCP": FeatureMetadata(
                name="Precipitation",
                description="Daily precipitation accumulation in mm",
                rationale="Heavy rain saturates soil, increases tree fall risk, causes flooding that damages electrical equipment",
                outage_relevance="Direct correlation with vegetation-related outages and equipment flooding",
                data_type="continuous",
            ),
            "WSF2": FeatureMetadata(
                name="Fastest 2-minute Wind Speed",
                description="Maximum sustained wind speed over 2-minute period in m/s",
                rationale="Wind is the primary cause of power line failures through tree contact and direct line damage",
                outage_relevance="Strong predictor of transmission and distribution system failures",
                data_type="continuous",
            ),
            # TODO: I will need to add more features here, also add the feature metadata for each feature
        }

        return base_features

    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """
        Organize features into logical groups for analysis and model interpretation
        This helps identify which types of weather patterns drive outages in different regions
        """
        return {
            "temperature_stress": [
                "TMAX",
                "TMIN",
                "temp_range_daily",
                "thermal_stress_index",
                "heat_wave",
                "extreme_heat",
                "freezing",
                "extreme_cold",
            ],
            "precipitation_impact": [
                "PRCP",
                "heavy_rain",
                "extreme_rain",
                "PRCP_7d_sum",
                "wet_period_indicator",
                "days_since_rain",
            ],
            "wind_damage": [
                "WSF2",
                "high_winds",
                "damaging_winds",
                "WSF2_7d_max",
                "mechanical_stress_index",
                "wind_acceleration_1d",
            ],
            "compound_risks": [
                "ice_storm_risk",
                "wet_windy_combo",
                "multiple_extremes",
                "weather_severity_score",
            ],
            "temporal_patterns": [
                "season",
                "month",
                "consecutive_heat_days",
                "consecutive_freeze_days",
            ],
        }
