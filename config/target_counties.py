
# Configuration for scaled power outage prediction data collection
# I'm defining target counties across diverse climate zones and infrastructure types

from typing import Dict, List, NamedTuple


class CountyConfig(NamedTuple):
    """
    I use this structure to maintain consistent county configuration
    Each county represents different outage risk profiles and weather patterns
    """

    fips_code: str
    name: str
    state: str
    climate_zone: str
    outage_risk_profile: str
    population: int
    notes: str


# I selected these counties to represent diverse outage scenarios across the US
# This geographic diversity ensures our model generalizes across different infrastructure and weather patterns
TARGET_COUNTIES: List[CountyConfig] = [
    # Texas - Winter storm and extreme heat vulnerabilities
    CountyConfig(
        fips_code="FIPS:48201",
        name="Harris County",
        state="TX",
        climate_zone="humid_subtropical",
        outage_risk_profile="winter_storms_heat_waves",
        population=4731145,
        notes="Houston metro - major urban center, ERCOT grid, Hurricane and winter storm exposure",
    ),
    CountyConfig(
        fips_code="FIPS:48029",
        name="Bexar County",
        state="TX",
        climate_zone="humid_subtropical",
        outage_risk_profile="extreme_heat_drought",
        population=2009324,
        notes="San Antonio - ERCOT grid, extreme heat events, large military infrastructure",
    ),
    # Florida - Hurricane and thunderstorm risks
    CountyConfig(
        fips_code="FIPS:12086",
        name="Miami-Dade County",
        state="FL",
        climate_zone="tropical",
        outage_risk_profile="hurricanes_thunderstorms",
        population=2701767,
        notes="Hurricane alley, high thunderstorm activity, aging coastal infrastructure",
    ),
    CountyConfig(
        fips_code="FIPS:12095",
        name="Orange County",
        state="FL",
        climate_zone="humid_subtropical",
        outage_risk_profile="hurricanes_lightning",
        population=1429908,
        notes="Orlando area - major tourism infrastructure, frequent lightning strikes",
    ),
    # Virginia - Ice storms and severe weather
    CountyConfig(
        fips_code="FIPS:51059",
        name="Fairfax County",
        state="VA",
        climate_zone="humid_subtropical",
        outage_risk_profile="ice_storms_severe_weather",
        population=1150309,
        notes="DC metro area - ice storm corridor, high infrastructure density",
    ),
    # California - Wildfires and Public Safety Power Shutoffs
    CountyConfig(
        fips_code="FIPS:06073",
        name="San Diego County",
        state="CA",
        climate_zone="mediterranean",
        outage_risk_profile="wildfires_psps",
        population=3298634,
        notes="Wildfire risk, Santa Ana winds, proactive power shutoffs (PSPS)",
    ),
    CountyConfig(
        fips_code="FIPS:06075",
        name="San Francisco County",
        state="CA",
        climate_zone="mediterranean",
        outage_risk_profile="earthquakes_fog",
        population=873965,
        notes="Seismic risk, underground infrastructure, unique microclimates",
    ),
    # Northeast - Winter storms and aging infrastructure
    CountyConfig(
        fips_code="FIPS:25025",
        name="Suffolk County",
        state="MA",
        climate_zone="humid_continental",
        outage_risk_profile="winter_storms_coastal",
        population=797936,
        notes="Boston area - nor'easters, coastal flooding, aging grid infrastructure",
    ),
    CountyConfig(
        fips_code="FIPS:36061",
        name="New York County",
        state="NY",
        climate_zone="humid_subtropical",
        outage_risk_profile="storms_high_density",
        population=1629153,
        notes="Manhattan - extremely high density, underground infrastructure, heat island effects",
    ),
    # Midwest - Severe thunderstorms and tornadoes
    CountyConfig(
        fips_code="FIPS:17031",
        name="Cook County",
        state="IL",
        climate_zone="humid_continental",
        outage_risk_profile="thunderstorms_winter",
        population=5275541,
        notes="Chicago area - severe thunderstorms, winter storms, major industrial load",
    ),
    # Southeast - Hurricane and ice storm combination
    CountyConfig(
        fips_code="FIPS:13121",
        name="Fulton County",
        state="GA",
        climate_zone="humid_subtropical",
        outage_risk_profile="ice_storms_thunderstorms",
        population=1066710,
        notes="Atlanta area - ice storm vulnerability, severe thunderstorms, rapid growth",
    ),
    # Pacific Northwest - Wind storms and flooding
    CountyConfig(
        fips_code="FIPS:53033",
        name="King County",
        state="WA",
        climate_zone="oceanic",
        outage_risk_profile="wind_storms_flooding",
        population=2269675,
        notes="Seattle area - Pacific storms, wind events, hydroelectric dependent",
    ),
]

# Define the 5-year collection window to capture multiple weather cycles
# This timeframe includes various El Niño/La Niña cycles and extreme weather events
DATA_COLLECTION_CONFIG = {
    "start_year": 2019,
    "end_year": 2024,
    "dataset_id": "GHCND",  # I use Global Historical Climatology Network Daily
    "base_datatypes": ["PRCP", "TMAX", "TMIN", "WSF2"],  # Core weather variables
    "extended_datatypes": [
        "AWND",
        "WSF5",
        "WT01",
        "WT02",
        "WT03",
        "WT04",
        "WT05",
        "WT06",
        "WT08",
        "WT11",
    ],
    "units": "metric",
}

# Include these additional weather types for enhanced feature engineering
EXTENDED_WEATHER_TYPES = {
    "AWND": "Average wind speed - I include this for sustained wind analysis beyond peak gusts",
    "WSF5": "Fastest 5-second wind speed - I use this to capture wind gusts that damage infrastructure",
    "WT01": "Fog, ice fog, or freezing fog - I track this as it affects visibility and infrastructure",
    "WT02": "Heavy fog or heaving freezing fog - I include this for severe visibility conditions",
    "WT03": "Thunder - I use this to identify thunderstorm activity beyond precipitation",
    "WT04": "Ice pellets, sleet, snow pellets, or small hail - I track this for infrastructure damage",
    "WT05": "Hail (may include small hail) - I include this as it directly damages power equipment",
    "WT06": "Glaze or rime - I track this for ice accumulation on power lines",
    "WT08": "Smoke or haze - I include this for wildfire and air quality indicators",
    "WT11": "High or damaging winds - I use this as a categorical indicator of damaging wind events",
}


def get_county_by_fips(fips_code: str) -> CountyConfig:
    """
    Provide this helper function to retrieve county configuration by FIPS code
    This maintains data integrity and provides easy access to county metadata
    """
    for county in TARGET_COUNTIES:
        if county.fips_code == fips_code:
            return county
    raise ValueError(f"County with FIPS code {fips_code} not found in configuration")


def get_counties_by_risk_profile(risk_profile: str) -> List[CountyConfig]:
    """
    Implement this to group counties by similar outage risk patterns
    This enables targeted analysis of specific weather-outage relationships
    """
    return [
        county
        for county in TARGET_COUNTIES
        if risk_profile in county.outage_risk_profile
    ]
