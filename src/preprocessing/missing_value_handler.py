"""
Missing value imputation for weather and outage data.

Architecture:
    - MissingValueAnalyzer: Diagnoses missingness patterns
    - ImputationStrategy: Abstract base for imputation methods
    - MissingValueHandler: Orchestrates imputation pipeline
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


@dataclass
class MissingnessReport:
    total_rows: int
    total_columns: int
    columns_with_missing: int
    missing_by_column: pd.DataFrame
    missing_patterns: pd.DataFrame

    def summary(self) -> str:
        pct_cols = 100 * self.columns_with_missing / self.total_columns
        lines = [
            f"Total rows: {self.total_rows:,}",
            f"Total columns: {self.total_columns}",
            f"Columns with missing: {self.columns_with_missing} ({pct_cols:.1f}%)",
            "",
            "Top 10 columns by missing count:",
            self.missing_by_column.head(10).to_string(),
        ]
        return "\n".join(lines)


class MissingValueAnalyzer:
    """Analyzes and reports missing value patterns in the dataset."""

    def analyze(self, df: pd.DataFrame) -> MissingnessReport:
        missing_counts = df.isnull().sum()
        missing_by_col = pd.DataFrame(
            {
                "column": missing_counts.index,
                "missing_count": missing_counts.values,
                "missing_pct": 100 * missing_counts.values / len(df),
            }
        )
        missing_by_col = missing_by_col[missing_by_col["missing_count"] > 0]
        missing_by_col = missing_by_col.sort_values("missing_count", ascending=False)

        missing_patterns = self._identify_patterns(df, missing_by_col)

        return MissingnessReport(
            total_rows=len(df),
            total_columns=len(df.columns),
            columns_with_missing=len(missing_by_col),
            missing_by_column=missing_by_col.reset_index(drop=True),
            missing_patterns=missing_patterns,
        )

    def _identify_patterns(
        self, df: pd.DataFrame, missing_by_col: pd.DataFrame
    ) -> pd.DataFrame:
        if missing_by_col.empty:
            return pd.DataFrame(columns=["pattern", "columns", "count"])

        patterns = []

        wt_cols = [c for c in df.columns if c.startswith("WT")]
        if wt_cols:
            wt_missing = df[wt_cols].isnull().all(axis=1).sum()
            patterns.append(
                {
                    "pattern": "Weather type codes (WT*) all missing",
                    "columns": (
                        ", ".join(wt_cols[:3]) + "..."
                        if len(wt_cols) > 3
                        else ", ".join(wt_cols)
                    ),
                    "count": wt_missing,
                }
            )

        wind_cols = [c for c in df.columns if "WSF" in c or c == "AWND"]
        if wind_cols:
            wind_missing = df[wind_cols].isnull().any(axis=1).sum()
            patterns.append(
                {
                    "pattern": "Wind measurements missing",
                    "columns": ", ".join(wind_cols[:5]),
                    "count": wind_missing,
                }
            )

        outage_cols = [
            c
            for c in df.columns
            if any(x in c for x in ["any_out", "customers_out", "minutes_out"])
        ]
        if outage_cols:
            outage_missing = df[outage_cols].isnull().any(axis=1).sum()
            patterns.append(
                {
                    "pattern": "Outage labels missing",
                    "columns": ", ".join(outage_cols[:5]),
                    "count": outage_missing,
                }
            )

        return pd.DataFrame(patterns)


class ImputationStrategy(ABC):
    """Abstract base class for imputation strategies."""

    @abstractmethod
    def fit(self, df: pd.DataFrame, columns: List[str]) -> "ImputationStrategy":
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        pass

    def fit_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        self.fit(df, columns)
        return self.transform(df, columns)


class ConstantImputation(ImputationStrategy):
    """Fill missing values with a constant."""

    def __init__(self, fill_value: float = 0.0):
        self.fill_value = fill_value

    def fit(self, df: pd.DataFrame, columns: List[str]) -> "ConstantImputation":
        return self

    def transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        df = df.copy()
        for col in columns:
            if col in df.columns:
                df[col] = df[col].fillna(self.fill_value)
        return df


class ForwardFillImputation(ImputationStrategy):
    """Forward fill within groups (e.g., by county)."""

    def __init__(self, group_cols: Optional[List[str]] = None):
        self.group_cols = group_cols or ["county_fips"]

    def fit(self, df: pd.DataFrame, columns: List[str]) -> "ForwardFillImputation":
        return self

    def transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        df = df.copy()
        group_cols_present = [c for c in self.group_cols if c in df.columns]

        if not group_cols_present:
            for col in columns:
                if col in df.columns:
                    df[col] = df[col].ffill()
        else:
            for col in columns:
                if col in df.columns:
                    df[col] = df.groupby(group_cols_present)[col].ffill()

        return df


class InterpolationImputation(ImputationStrategy):
    """Linear interpolation within groups."""

    def __init__(self, group_cols: Optional[List[str]] = None, method: str = "linear"):
        self.group_cols = group_cols or ["county_fips"]
        self.method = method

    def fit(self, df: pd.DataFrame, columns: List[str]) -> "InterpolationImputation":
        return self

    def transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        df = df.copy()
        group_cols_present = [c for c in self.group_cols if c in df.columns]

        if not group_cols_present:
            for col in columns:
                if col in df.columns:
                    df[col] = df[col].interpolate(method=self.method)
        else:
            for col in columns:
                if col in df.columns:
                    df[col] = df.groupby(group_cols_present)[col].transform(
                        lambda x: x.interpolate(method=self.method)
                    )

        return df


class MeanImputation(ImputationStrategy):
    """Fill with mean value, optionally by group."""

    def __init__(self, group_cols: Optional[List[str]] = None):
        self.group_cols = group_cols
        self.means: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame, columns: List[str]) -> "MeanImputation":
        for col in columns:
            if col in df.columns:
                if self.group_cols:
                    self.means[col] = df.groupby(self.group_cols)[col].transform("mean")
                else:
                    self.means[col] = df[col].mean()
        return self

    def transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        df = df.copy()
        for col in columns:
            if col in df.columns and col in self.means:
                if isinstance(self.means[col], pd.Series):
                    df[col] = df[col].fillna(self.means[col])
                else:
                    df[col] = df[col].fillna(self.means[col])
        return df


class MissingValueHandler:
    """
    Orchestrates missing value imputation using domain-specific strategies.

    Strategy assignment:
        - Weather type codes (WT*): 0 (event didn't occur)
        - Base weather (PRCP): 0 (no precipitation)
        - Temperature (TMAX, TMIN): Linear interpolation
        - Wind (WSF2, AWND): Forward fill, then mean
        - Derived features: Recompute after base imputation
        - Outage labels: 0 (no outage observed)
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.analyzer = MissingValueAnalyzer()
        self.strategies: Dict[str, Tuple[ImputationStrategy, List[str]]] = {}

    def analyze(self, df: pd.DataFrame) -> MissingnessReport:
        """Generate missingness report."""
        report = self.analyzer.analyze(df)
        if self.verbose:
            print(report.summary())
        return report

    def fit(self, df: pd.DataFrame) -> "MissingValueHandler":
        """
        Learn imputation parameters from training data.
        Define strategies for different feature groups.
        """
        self._define_strategies(df)

        for name, (strategy, columns) in self.strategies.items():
            cols_present = [c for c in columns if c in df.columns]
            if cols_present:
                strategy.fit(df, cols_present)
                if self.verbose:
                    print(f"Fitted strategy '{name}' for {len(cols_present)} columns")

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply learned imputation strategies."""
        df = df.copy()

        for name, (strategy, columns) in self.strategies.items():
            cols_present = [c for c in columns if c in df.columns]
            if cols_present:
                df = strategy.transform(df, cols_present)
                if self.verbose:
                    remaining = df[cols_present].isnull().sum().sum()
                    print(
                        f"Applied '{name}': {remaining} missing values remaining in group"
                    )

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)

    def _define_strategies(self, df: pd.DataFrame) -> None:
        """Define imputation strategies for different feature groups."""

        wt_cols = [c for c in df.columns if c.startswith("WT")]
        self.strategies["weather_type_codes"] = (
            ConstantImputation(fill_value=0.0),
            wt_cols,
        )

        prcp_cols = ["PRCP"] + [c for c in df.columns if c.startswith("PRCP_lag")]
        self.strategies["precipitation"] = (
            ConstantImputation(fill_value=0.0),
            prcp_cols,
        )

        temp_cols = ["TMAX", "TMIN"] + [
            c
            for c in df.columns
            if c.startswith("TMAX_lag") or c.startswith("TMIN_lag")
        ]
        self.strategies["temperature"] = (
            InterpolationImputation(group_cols=["county_fips"], method="linear"),
            temp_cols,
        )

        wind_base = ["WSF2", "WSF5", "AWND"]
        wind_derived = [
            c for c in df.columns if any(x in c for x in ["WSF2_", "WSF5_", "AWND_"])
        ]
        self.strategies["wind_base"] = (
            ForwardFillImputation(group_cols=["county_fips"]),
            wind_base,
        )
        self.strategies["wind_derived"] = (
            MeanImputation(group_cols=["county_fips"]),
            wind_derived,
        )

        binary_features = [
            "heavy_rain",
            "extreme_rain",
            "heat_wave",
            "extreme_heat",
            "freezing",
            "extreme_cold",
            "high_winds",
            "damaging_winds",
            "winter_freeze",
            "summer_heat",
            "spring_storms",
            "ice_storm_risk",
            "wet_windy_combo",
            "heat_demand_stress",
            "light_rain",
            "moderate_rain",
        ]
        self.strategies["binary_indicators"] = (
            ConstantImputation(fill_value=0.0),
            binary_features,
        )

        outage_numeric = [
            "any_out",
            "num_out_per_day",
            "minutes_out",
            "customers_out",
            "customers_out_mean",
            "cust_minute_area",
            "pct_out_max",
            "pct_out_area",
        ]
        self.strategies["outage_labels"] = (
            ConstantImputation(fill_value=0.0),
            outage_numeric,
        )

        continuous_features = [
            "temp_range_daily",
            "temp_volatility_3d",
            "temp_change_1d",
            "heating_degree_days",
            "cooling_degree_days",
            "days_since_rain",
            "precip_volatility_7d",
            "wind_acceleration_1d",
        ]
        self.strategies["continuous_derived"] = (
            ForwardFillImputation(group_cols=["county_fips"]),
            continuous_features,
        )

        stress_indices = [
            "thermal_stress_index",
            "mechanical_stress_index",
            "weather_severity_score",
        ]
        self.strategies["stress_indices"] = (
            MeanImputation(group_cols=["county_fips"]),
            stress_indices,
        )


def create_missing_value_handler(verbose: bool = True) -> MissingValueHandler:
    """Factory function to create a configured handler."""
    return MissingValueHandler(verbose=verbose)


if __name__ == "__main__":
    """
    for testing.
    """
    print("Loading merged dataset...")
    df = pd.read_csv(
        "data/processed/merged_weather_outages_2019_2024_keep_all.csv", low_memory=False
    )
    print("\n=== Initial Analysis ===")
    handler = create_missing_value_handler(verbose=True)
    report = handler.analyze(df)

    print("\n=== Applying Imputation ===")
    df_imputed = handler.fit_transform(df)
    print("\n=== Post-Imputation Analysis ===")
    report_after = handler.analyze(df_imputed)
    print("\n=== Saving Result ===")
    output_path = "data/ml_ready/merged_weather_outages_2019_2024_imputed.csv"
    df_imputed.to_csv(output_path, index=False)
    print(f"Saved imputed dataset to: {output_path}")
