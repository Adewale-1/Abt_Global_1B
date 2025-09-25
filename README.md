
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
