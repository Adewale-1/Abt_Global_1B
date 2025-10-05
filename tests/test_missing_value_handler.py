"""
Unit tests for missing value handler.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "preprocessing"))

from missing_value_handler import (
    MissingValueAnalyzer,
    ConstantImputation,
    ForwardFillImputation,
    InterpolationImputation,
    MeanImputation,
    MissingValueHandler,
    create_missing_value_handler,
)


@pytest.fixture
def sample_weather_data():
    """Create sample weather data with known missing patterns."""
    np.random.seed(42)
    n_rows = 100
    data = {
        "day": pd.date_range("2021-01-01", periods=n_rows, freq="D"),
        "county_fips": ["06073"] * 50 + ["48201"] * 50,
        "PRCP": np.random.uniform(0, 50, n_rows),
        "TMAX": np.random.uniform(10, 35, n_rows),
        "TMIN": np.random.uniform(-5, 20, n_rows),
        "WSF2": np.random.uniform(0, 30, n_rows),
        "WT01": np.random.choice([0, 1, np.nan], n_rows, p=[0.7, 0.1, 0.2]),
        "WT03": np.random.choice([0, 1, np.nan], n_rows, p=[0.8, 0.05, 0.15]),
        "any_out": np.random.choice([0, 1], n_rows, p=[0.1, 0.9]),
        "customers_out": np.random.uniform(0, 5000, n_rows),
    }
    df = pd.DataFrame(data)
    df.loc[10:15, "PRCP"] = np.nan
    df.loc[20:22, "TMAX"] = np.nan
    df.loc[30:35, "WSF2"] = np.nan
    df.loc[90:95, "any_out"] = np.nan

    return df


class TestMissingValueAnalyzer:

    def test_analyze_no_missing(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        analyzer = MissingValueAnalyzer()
        report = analyzer.analyze(df)
        assert report.total_rows == 3
        assert report.total_columns == 2
        assert report.columns_with_missing == 0

    def test_analyze_with_missing(self, sample_weather_data):
        analyzer = MissingValueAnalyzer()
        report = analyzer.analyze(sample_weather_data)
        assert report.total_rows == 100
        assert report.columns_with_missing > 0
        assert "PRCP" in report.missing_by_column["column"].values

    def test_identify_patterns(self, sample_weather_data):
        analyzer = MissingValueAnalyzer()
        report = analyzer.analyze(sample_weather_data)
        assert not report.missing_patterns.empty
        pattern_names = report.missing_patterns["pattern"].values
        assert any("Weather type" in p for p in pattern_names)


class TestConstantImputation:

    def test_constant_fill_zero(self):
        df = pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, 5, 6]})
        strategy = ConstantImputation(fill_value=0.0)
        result = strategy.fit_transform(df, ["a", "b"])

        assert result["a"].isnull().sum() == 0
        assert result["b"].isnull().sum() == 0
        assert result.loc[1, "a"] == 0.0
        assert result.loc[0, "b"] == 0.0

    def test_constant_fill_custom_value(self):
        df = pd.DataFrame({"a": [1, np.nan, 3]})
        strategy = ConstantImputation(fill_value=-999)
        result = strategy.fit_transform(df, ["a"])

        assert result.loc[1, "a"] == -999


class TestForwardFillImputation:

    def test_forward_fill_no_groups(self):
        df = pd.DataFrame({"a": [1, np.nan, np.nan, 4, np.nan]})
        strategy = ForwardFillImputation(group_cols=[])
        result = strategy.fit_transform(df, ["a"])

        assert result["a"].tolist() == [1, 1, 1, 4, 4]

    def test_forward_fill_with_groups(self):
        df = pd.DataFrame(
            {"county": ["A", "A", "B", "B"], "value": [1, np.nan, 10, np.nan]}
        )
        strategy = ForwardFillImputation(group_cols=["county"])
        result = strategy.fit_transform(df, ["value"])

        assert result.loc[1, "value"] == 1.0
        assert result.loc[3, "value"] == 10.0


class TestInterpolationImputation:

    def test_linear_interpolation(self):
        df = pd.DataFrame(
            {"county": ["A"] * 5, "temp": [10, np.nan, np.nan, 20, np.nan]}
        )
        strategy = InterpolationImputation(group_cols=["county"], method="linear")
        result = strategy.fit_transform(df, ["temp"])

        assert np.isclose(result.loc[1, "temp"], 12.5, atol=0.1)
        assert np.isclose(result.loc[2, "temp"], 15.0, atol=0.1)


class TestMeanImputation:

    def test_mean_no_groups(self):
        df = pd.DataFrame({"a": [10, np.nan, 30, np.nan]})
        strategy = MeanImputation(group_cols=None)
        result = strategy.fit_transform(df, ["a"])

        assert result["a"].isnull().sum() == 0
        assert np.isclose(result.loc[1, "a"], 20.0)

    def test_mean_with_groups(self):
        df = pd.DataFrame(
            {"group": ["A", "A", "B", "B"], "value": [10, np.nan, 100, np.nan]}
        )
        strategy = MeanImputation(group_cols=["group"])
        result = strategy.fit_transform(df, ["value"])

        assert np.isclose(result.loc[1, "value"], 10.0)
        assert np.isclose(result.loc[3, "value"], 100.0)


class TestMissingValueHandler:

    def test_handler_reduces_missing(self, sample_weather_data):
        initial_missing = sample_weather_data.isnull().sum().sum()
        handler = MissingValueHandler(verbose=False)
        result = handler.fit_transform(sample_weather_data)
        final_missing = result.isnull().sum().sum()
        assert final_missing < initial_missing

    def test_handler_preserves_shape(self, sample_weather_data):
        handler = MissingValueHandler(verbose=False)
        result = handler.fit_transform(sample_weather_data)
        assert result.shape == sample_weather_data.shape

    def test_handler_fit_transform_separate(self, sample_weather_data):
        handler = MissingValueHandler(verbose=False)
        handler.fit(sample_weather_data)
        result = handler.transform(sample_weather_data)
        assert result.shape == sample_weather_data.shape

    def test_factory_function(self):
        handler = create_missing_value_handler(verbose=False)
        assert isinstance(handler, MissingValueHandler)
        assert handler.verbose is False


class TestEndToEnd:

    def test_full_pipeline(self, sample_weather_data):
        """Test complete missing value handling pipeline."""
        initial_shape = sample_weather_data.shape
        initial_missing = sample_weather_data.isnull().sum().sum()
        handler = create_missing_value_handler(verbose=False)
        report_before = handler.analyze(sample_weather_data)
        assert report_before.columns_with_missing > 0
        result = handler.fit_transform(sample_weather_data)
        assert result.shape == initial_shape
        final_missing = result.isnull().sum().sum()
        assert final_missing < initial_missing
        report_after = handler.analyzer.analyze(result)
        assert report_after.columns_with_missing <= report_before.columns_with_missing


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
