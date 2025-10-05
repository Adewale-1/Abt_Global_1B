## Quick Start

### 1. Environment Setup

First, obtain a NOAA Climate Data Online API token:

1. Visit https://www.ncei.noaa.gov/cdo-web/token
2. Register for a free token
3. Create `.env` file and add your token:

```
NOAA_CDO_TOKEN=your_token_here
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Data Processor

#### Complete Workflow (Recommended)

```bash
cd src/data_collection
python main_dataprocessor.py
```

#### Custom Date Range

```bash
python main_dataprocessor.py --start-year 2020 --end-year 2024
```

#### Validate Environment Only

```bash
python main_dataprocessor.py --validate-only
```

## What It Does

The main data processor will:

1. **Collect Weather Data**: Download 5 years of weather data for 12 diverse U.S. counties
2. **Create Unified Dataset**: Combine all county data into a single structured dataset
3. **Engineer Features**: Generate 70+ ML features from raw weather data
4. **Generate Reports**: Create summary reports and data quality analysis

## Output Structure

After successful processing, you'll have:

```
raw/                    # Original weather station data
processed/             # Unified county datasets
ml_ready/              # ML-ready features and reports
```

## Target Counties

The processor collects data for these strategically selected counties:

- **Texas**: Harris County (Houston), Bexar County (San Antonio)
- **Florida**: Miami-Dade County, Orange County (Orlando)
- **California**: San Diego County, San Francisco County
- **Northeast**: Suffolk County (Boston), New York County (Manhattan)
- **Other**: Fairfax County (VA), Cook County (IL), Fulton County (GA), King County (WA)

## Features Generated

The processor creates features targeting different outage causes:

- **Temperature Stress**: Thermal cycling, degree days, heat waves, freeze cycles
- **Precipitation Impact**: Accumulation patterns, intensity, soil saturation indicators
- **Wind Damage**: Speed categories, sustained winds, acceleration, fatigue factors
- **Compound Risks**: Ice storms, wet/windy combinations, multiple hazards
- **Temporal Patterns**: Seasonal effects, consecutive extreme days

## Data Dictionary & Abbreviations

### Core Weather Measurements (NOAA CDO)

| **Abbreviation** | **Full Name**         | **Units**   | **Description**                                 |
| ---------------- | --------------------- | ----------- | ----------------------------------------------- |
| **AWND**         | Average Wind Speed    | mph         | Daily average wind speed                        |
| **PRCP**         | Precipitation         | 0.01 inches | Total daily precipitation                       |
| **TMAX**         | Maximum Temperature   | °C          | Daily maximum temperature                       |
| **TMIN**         | Minimum Temperature   | °C          | Daily minimum temperature                       |
| **WSF2**         | Fastest 2-Minute Wind | mph         | Peak sustained wind speed (power grid critical) |
| **WSF5**         | Fastest 5-Second Wind | mph         | Peak wind gust                                  |

### Weather Type Codes (WT01-WT11)

| **Code** | **Weather Type**                    | **Power Outage Risk** | **Description**                             |
| -------- | ----------------------------------- | --------------------- | ------------------------------------------- |
| **WT01** | Fog/Visibility ≤ 0.25 miles         | Low                   | Reduced visibility, minor transport issues  |
| **WT02** | Heavy Fog/Visibility ≤ 0.0625 miles | Medium                | Severe visibility reduction                 |
| **WT03** | Thunder                             | High                  | Lightning strikes (direct equipment damage) |
| **WT04** | Ice pellets/sleet                   | **Very High**         | Ice loading on power lines                  |
| **WT05** | Hail                                | High                  | Physical equipment damage                   |
| **WT06** | Glaze/rime ice                      | **Very High**         | Line icing (critical for power grid)        |
| **WT08** | Smoke/haze                          | Low                   | Air quality issues                          |
| **WT11** | High/damaging winds                 | **Very High**         | Tree contact, line breaks                   |

### Temporal Feature Naming Convention

| **Pattern**                     | **Meaning**      | **Example**                        | **Purpose**                  |
| ------------------------------- | ---------------- | ---------------------------------- | ---------------------------- |
| **\_lag1d, \_lag2d, \_lag3d**   | 1, 2, 3 days ago | `PRCP_lag1d` = yesterday's rain    | Capture weather persistence  |
| **\_3d, \_7d, \_14d**           | Rolling windows  | `TMAX_7d_mean` = week average temp | Identify trends and patterns |
| **\_sum, \_mean, \_max, \_min** | Aggregation type | `PRCP_3d_sum` = 3-day total rain   | Different impact measures    |

### Risk Indicator Features (Binary 0/1)

| **Feature**                 | **Threshold/Logic**         | **Outage Risk** | **Rationale**                         |
| --------------------------- | --------------------------- | --------------- | ------------------------------------- |
| **heavy_rain**              | PRCP > 25mm/day             | Medium          | Flooding, saturated soil around poles |
| **extreme_rain**            | PRCP > 50mm/day             | **High**        | Flash floods, washouts                |
| **heat_wave**               | 3+ consecutive hot days     | High            | Sustained demand stress               |
| **extreme_heat**            | TMAX > 35°C (95°F)          | **Very High**   | Peak demand, equipment overheating    |
| **freezing**                | TMIN < 0°C (32°F)           | High            | Ice formation on lines                |
| **extreme_cold**            | TMIN < -10°C (14°F)         | **Very High**   | Brittle lines, thermal stress         |
| **high_winds**              | WSF2 > 39 mph               | High            | Tree contact with lines               |
| **damaging_winds**          | WSF2 > 58 mph               | **Very High**   | Direct line damage                    |
| **consecutive_heat_days**   | Count of heat wave duration | High            | Cumulative thermal stress             |
| **consecutive_freeze_days** | Count of freeze duration    | High            | Ice accumulation                      |
| **consecutive_rain_days**   | Count of wet period         | Medium          | Soil saturation, tree instability     |

### Compound Risk Features

| **Feature**                 | **Description**        | **Power Outage Impact** | **Formula/Logic**                     |
| --------------------------- | ---------------------- | ----------------------- | ------------------------------------- |
| **ice_storm_risk**          | Freezing rain + wind   | **Critical**            | Ice formation during windy conditions |
| **wet_windy_combo**         | Heavy rain + high wind | **High**                | Saturated soil + wind throw trees     |
| **thermal_stress_index**    | Temperature volatility | High                    | Daily temp range + multi-day changes  |
| **mechanical_stress_index** | Wind + ice loading     | **Critical**            | Combined physical stresses            |
| **weather_severity_score**  | Overall daily severity | Variable                | Composite risk assessment             |
| **multiple_extremes**       | 2+ extreme conditions  | **High**                | Multiplicative risk effects           |

### Derived Meteorological Features

| **Feature**              | **Description**              | **Units** | **Outage Relevance**          |
| ------------------------ | ---------------------------- | --------- | ----------------------------- |
| **heating_degree_days**  | Energy demand for heating    | °C-days   | Winter demand stress          |
| **cooling_degree_days**  | Energy demand for cooling    | °C-days   | Summer demand stress          |
| **temp_range_daily**     | TMAX - TMIN                  | °C        | Thermal expansion/contraction |
| **temp_volatility_3d**   | 3-day temperature variance   | °C²       | Equipment stress cycles       |
| **freeze_thaw_cycle**    | Temperature crosses 0°C      | Binary    | Line expansion/contraction    |
| **days_since_rain**      | Drought/dry period indicator | Days      | Fire risk, demand patterns    |
| **wind_acceleration_1d** | Wind speed change rate       | mph/day   | Dynamic loading stress        |

### Categorical Variables

| **Variable**            | **Categories**                                                            | **Use Case**                      |
| ----------------------- | ------------------------------------------------------------------------- | --------------------------------- |
| **season**              | winter, spring, summer, fall                                              | Seasonal outage patterns          |
| **climate_zone**        | mediterranean, tropical, humid_subtropical, humid_continental, oceanic    | Regional risk profiles            |
| **outage_risk_profile** | wildfires_psps, hurricanes_thunderstorms, ice_storms_severe_weather, etc. | County-specific risk patterns     |
| **state**               | CA, FL, GA, IL, MA, NY, TX, VA, WA                                        | Geographic regulatory differences |

### Geographic Identifiers

| **Variable**    | **Description**                              | **Example**               |
| --------------- | -------------------------------------------- | ------------------------- |
| **county_fips** | Federal Information Processing Standard code | 48201 (Harris County, TX) |
| **county_name** | Full county name                             | "Harris County"           |
| **state**       | Two-letter state code                        | "TX"                      |


# Preprocessing Module

## Preprocessing Step 1: Missing Value Handling

### Components

#### 1. MissingValueAnalyzer

Diagnoses missing data patterns and generates reports.

#### 2. ImputationStrategy (Abstract Base Class)

- `ConstantImputation`: Fill with constant value (0 for binary/precipitation)
- `ForwardFillImputation`: Carry forward last observation within groups
- `InterpolationImputation`: Linear interpolation for smooth temporal features
- `MeanImputation`: Fill with group-wise or global mean

#### 3. MissingValueHandler

Orchestrates imputation pipeline with domain-specific strategies for different feature types.

### Imputation Strategy by Feature Type

| Feature Group                  | Strategy             | Columns Affected  | Rationale                             |
| ------------------------------ | -------------------- | ----------------- | ------------------------------------- |
| Weather type codes (WT01-WT11) | Constant (0)         | 8 columns         | Missing indicates event did not occur |
| Precipitation                  | Constant (0)         | PRCP, PRCP_lag\*  | No precipitation recorded             |
| Temperature                    | Linear interpolation | TMAX, TMIN, lags  | Smooth temporal variation             |
| Wind base                      | Forward fill         | WSF2, WSF5, AWND  | Carry forward recent conditions       |
| Wind derived                   | Group mean           | WSF2_3d_max, etc. | Complex derived features              |
| Binary indicators              | Constant (0)         | 16 features       | Absence of extreme event              |
| Outage labels                  | Constant (0)         | 8 features        | No outage observed                    |
| Continuous derived             | Forward fill         | 8 features        | Maintain temporal continuity          |
| Stress indices                 | Group mean           | 3 features        | Complex derived metrics               |

## Results

### Before Imputation

- Total missing values: **219,619**
- Columns with missing: **48 / 108 (44.4%)**
- Most affected:
  - WT05: 99.6% missing
  - WT11: 99.5% missing
  - WT04: 99.4% missing
  - Wind measurements: ~9% missing

### After Imputation

- Total missing values: **33,670**
- Columns with missing: **30 / 108 (27.8%)**
- Reduction: **84.7%**
- Remaining missing values primarily in wind measurements for counties without wind sensors

### Key Improvements

- Eliminated all missing values in weather type codes
- Eliminated all missing values in precipitation features
- Reduced temperature missing values by 99.4% (144 remaining)
- Reduced outage label missing values to 0
- Maintained data integrity with county-grouped imputation

## Usage

### Command Line

```bash
python src/preprocessing/run_missing_value_imputation.py \
    --input data/processed/merged_weather_outages_2019_2024_keep_all.csv \
    --output data/ml_ready/merged_weather_outages_2019_2024_imputed.csv \
    --report results/reports/missingness_report.csv
```


## Output Files Generated

1. `./data/ml_ready/merged_weather_outages_2019_2024_imputed.csv` - Full imputed dataset (25,993 rows × 108 columns)
2. `results/reports/missingness_report.csv` - Detailed missing value analysis by column