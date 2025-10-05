# EDA Report — merged_weather_outages_2019_2024_keep_all
- Generated: 2025-10-05T01:45:04
- Source: `C:\Users\Owner\.cursor\ABT Global\Abt_Global_1B\src\data_collection\data\processed\merged_weather_outages_2019_2024_keep_all.csv`

## 1) Dataset Overview
- Shape: **25,993 rows × 108 columns**
- Memory usage: **32.24 MB**
- Date column: `day` → **2019-01-01 00:00:00 → 2024-12-31 00:00:00**
- County count (fips_code): **12**

## 2) Dtypes Summary
- Dtype counts: {dtype('float64'): 63, dtype('int64'): 35, dtype('O'): 9, dtype('<M8[ns]'): 1}

## 3) Missingness
- Columns with missing values: **48 / 108**

Top missing columns (first 30):
|                         |   missing_count |
|:------------------------|----------------:|
| WT05                    |           25892 |
| WT11                    |           25870 |
| WT04                    |           25831 |
| WT06                    |           25795 |
| WT02                    |           23816 |
| WT03                    |           22024 |
| WT08                    |           20232 |
| WT01                    |           14114 |
| mechanical_stress_index |            2353 |
| wind_acceleration_1d    |            2353 |
| WSF5                    |            2351 |
| WSF2_lag2d              |            2348 |
| WSF2_lag1d              |            2337 |
| AWND                    |            2333 |
| WSF2                    |            2326 |
| WSF2_3d_max             |            2306 |
| WSF2_7d_max             |            2291 |
| WSF2_7d_mean            |            2291 |
| pct_out_area_covered    |            1132 |
| customers_out_mean      |            1115 |
| pct_out_max_unified     |            1115 |
| train_mask              |             996 |
| snapshot_minutes        |             996 |
| run_start_time_day      |             996 |
| minutes_in_local_day    |             996 |
| customers_total         |             996 |
| minutes_observed        |             996 |
| snapshots_count         |             996 |
| coverage                |             996 |
| pct_out_area_unified    |             996 |

## 4) Duplicates
- Exact duplicate rows: **0**
- Duplicate `(fips_code, day)` rows: **0**

## 5) Date Coverage by County
|   fips_code | first_date          | last_date           |   observed_days |   expected_days |   missing_days_est |
|------------:|:--------------------|:--------------------|----------------:|----------------:|-------------------:|
|       06073 | 2019-01-01 00:00:00 | 2024-12-31 00:00:00 |            1881 |            2192 |                311 |
|       06075 | 2019-01-01 00:00:00 | 2024-12-31 00:00:00 |            2192 |            2192 |                  0 |
|       12086 | 2019-01-01 00:00:00 | 2024-12-31 00:00:00 |            2192 |            2192 |                  0 |
|       12095 | 2019-01-01 00:00:00 | 2024-12-31 00:00:00 |            2192 |            2192 |                  0 |
|       13121 | 2019-01-01 00:00:00 | 2024-12-31 00:00:00 |            2192 |            2192 |                  0 |
|       17031 | 2019-01-01 00:00:00 | 2024-12-31 00:00:00 |            2192 |            2192 |                  0 |
|       25025 | 2019-01-01 00:00:00 | 2024-12-31 00:00:00 |            2192 |            2192 |                  0 |
|       36061 | 2019-01-01 00:00:00 | 2024-12-31 00:00:00 |            2192 |            2192 |                  0 |
|       48029 | 2019-01-01 00:00:00 | 2024-12-31 00:00:00 |            2192 |            2192 |                  0 |
|       48201 | 2019-01-01 00:00:00 | 2024-12-31 00:00:00 |            2192 |            2192 |                  0 |
|       51059 | 2019-01-01 00:00:00 | 2024-12-31 00:00:00 |            2192 |            2192 |                  0 |
|       53033 | 2019-01-01 00:00:00 | 2024-12-31 00:00:00 |            2192 |            2192 |                  0 |

## 6) Outage Labels / Diagnostics
- `any_out` distribution: {1: 24861, 0: 1132} (positives: **95.64%**)

Summary stats (labels/diagnostics):
|                      |   count |             mean |              std |             min |              25% |              50% |              75% |            max |
|:---------------------|--------:|-----------------:|-----------------:|----------------:|-----------------:|-----------------:|-----------------:|---------------:|
| any_out              |   25993 |      0.95645     |      0.204096    |      0          |      1           |      1           |      1           |    1           |
| num_out_per_day      |   25993 |      2.01043     |      1.4809      |      0          |      1           |      1           |      3           |   16           |
| minutes_out          |   25993 |   1133.73        |    424.729       |      0          |    990           |   1350           |   1440           | 1500           |
| customers_out        |   25993 |   2213.84        |  20101.2         |      0          |    141           |    606           |   1803           |    1.6607e+06  |
| cust_minute_area     |   25993 | 968916           |      2.11142e+07 |      0          |  32460           | 126705           | 330255           |    1.65193e+09 |
| pct_out_max          |   25993 |      0.00259285  |      0.0144951   |      0          |      0.000203667 |      0.000681189 |      0.00212531  |    0.908637    |
| pct_out_area         |   25993 |      0.000620797 |      0.00963814  |      0          |      3.1823e-05  |      0.000107436 |      0.000269487 |    0.627665    |
| customers_total      |   24997 | 976183           | 568442           | 209753          | 402618           | 951391           |      1.36293e+06 |    2.16201e+06 |
| coverage             |   24997 |      0.81868     |      0.254495    |      0          |      0.729167    |      0.947917    |      1           |    1           |
| minutes_in_local_day |   24997 |   1440.01        |      4.44196     |   1380          |   1440           |   1440           |   1440           | 1500           |
| snapshots_count      |   24997 |     78.5938      |     24.4336      |      0          |     70           |     91           |     96           |  100           |
| snapshot_minutes     |   24997 |     15           |      0           |     15          |     15           |     15           |     15           |   15           |
| customers_out_mean   |   24878 |    720.485       |  15001           |      0          |     37.1079      |    110.76        |    255.88        |    1.14718e+06 |
| pct_out_area_unified |   24997 |      0.000645533 |      0.00982748  |      0          |      3.82573e-05 |      0.000115224 |      0.000280171 |    0.627665    |
| pct_out_area_covered |   24861 |      0.000686589 |      0.010046    |      5.4714e-07 |      4.91345e-05 |      0.00013226  |      0.000312533 |    0.627665    |
| pct_out_max_unified  |   24878 |      0.00270906  |      0.0148057   |      0          |      0.000247911 |      0.000749973 |      0.00221506  |    0.908637    |

## 7) Weather Feature Presence
- Expected weather columns present: {'PRCP': True, 'TMAX': True, 'TMIN': True, 'WSF2': True, 'AWND': True, 'WSF5': True}

## 8) High Correlations (|r| ≥ 0.95)
| feature_1            | feature_2            |     corr |
|:---------------------|:---------------------|---------:|
| minutes_out          | minutes_observed     | 1        |
| pct_out_area         | pct_out_area_unified | 1        |
| pct_out_max          | pct_out_max_unified  | 1        |
| minutes_out          | snapshots_count      | 1        |
| snapshots_count      | minutes_observed     | 1        |
| customers_out_mean   | cust_minute_area     | 0.999966 |
| snapshots_count      | coverage             | 0.999952 |
| minutes_out          | coverage             | 0.999952 |
| minutes_observed     | coverage             | 0.999952 |
| pct_out_area         | pct_out_area_covered | 0.998201 |
| pct_out_area_unified | pct_out_area_covered | 0.998201 |
| month                | day_of_year          | 0.996497 |
| PRCP_7d_max          | precip_volatility_7d | 0.99613  |
| TMIN_lag1d           | TMIN_3d_mean         | 0.991591 |
| TMAX_lag1d           | TMAX_3d_mean         | 0.990194 |
| extreme_rain         | multiple_extremes    | 0.984172 |
| TMIN_7d_mean         | TMIN_7d_min          | 0.979336 |
| TMIN_3d_mean         | TMIN_7d_mean         | 0.975284 |
| TMAX_3d_mean         | TMAX_7d_mean         | 0.975108 |
| TMIN                 | TMIN_3d_mean         | 0.97416  |
| TMIN_lag2d           | TMIN_3d_mean         | 0.974145 |
| TMAX                 | TMAX_3d_mean         | 0.972319 |
| TMAX_lag2d           | TMAX_3d_mean         | 0.972301 |
| TMIN_7d_min          | TMIN_14d_min         | 0.972041 |
| TMAX_7d_mean         | TMAX_7d_max          | 0.97174  |
| high_winds           | strong_winds         | 0.967852 |
| TMIN_lag3d           | TMIN_7d_mean         | 0.967503 |
| TMIN_lag2d           | TMIN_7d_mean         | 0.96569  |
| TMAX_lag3d           | TMAX_7d_mean         | 0.965143 |
| TMAX_lag2d           | TMAX_7d_mean         | 0.963487 |
| TMAX_7d_max          | TMAX_14d_max         | 0.962435 |
| PRCP_3d_sum          | PRCP_3d_max          | 0.960367 |
| TMIN_lag1d           | TMIN_7d_mean         | 0.958565 |
| TMIN                 | TMIN_lag1d           | 0.957578 |
| TMIN_lag1d           | TMIN_lag2d           | 0.957576 |
| TMIN_lag2d           | TMIN_lag3d           | 0.957575 |
| TMAX_lag1d           | TMAX_7d_mean         | 0.956838 |
| TMIN_3d_mean         | TMIN_7d_min          | 0.955315 |
| TMAX_7d_mean         | TMIN_7d_mean         | 0.954944 |
| TMIN_7d_mean         | TMIN_14d_min         | 0.953667 |
| TMAX_lag2d           | TMAX_lag3d           | 0.953023 |
| TMAX_lag1d           | TMAX_lag2d           | 0.953017 |
| TMAX                 | TMAX_lag1d           | 0.953006 |

## 9) Feature–Target Correlations (top |r|)
### Target: `any_out`
| feature            |       corr |
|:-------------------|-----------:|
| minutes_out        |  0.569602  |
| num_out_per_day    |  0.289691  |
| year               |  0.253916  |
| coverage           |  0.237932  |
| minutes_observed   |  0.237913  |
| snapshots_count    |  0.237913  |
| WSF2_7d_mean       |  0.101037  |
| WSF5               |  0.0728799 |
| TMIN_14d_min       |  0.0707856 |
| WSF2               |  0.0685284 |
| TMIN_7d_min        |  0.0673648 |
| WSF2_lag1d         |  0.0618442 |
| WSF2_lag2d         |  0.0599894 |
| TMIN_7d_mean       |  0.0587009 |
| temp_volatility_3d | -0.0569471 |
| TMIN_lag3d         |  0.0548828 |
| TMIN_3d_mean       |  0.0548067 |
| TMIN_lag1d         |  0.053647  |
| TMAX_7d_mean       |  0.0534529 |
| TMIN_lag2d         |  0.0534109 |

### Target: `minutes_out`
| feature              |      corr |
|:---------------------|----------:|
| minutes_observed     |  1        |
| snapshots_count      |  1        |
| coverage             |  0.999952 |
| any_out              |  0.569602 |
| num_out_per_day      | -0.316224 |
| PRCP_14d_sum         |  0.284203 |
| WSF2_7d_mean         |  0.281138 |
| PRCP_7d_sum          |  0.242347 |
| wet_period_indicator |  0.236629 |
| customers_total      |  0.234685 |
| cooling_degree_days  |  0.232838 |
| PRCP_7d_max          |  0.231489 |
| precip_volatility_7d |  0.227946 |
| heavy_rain           |  0.223016 |
| extreme_rain         |  0.220748 |
| multiple_extremes    |  0.220127 |
| TMAX_7d_mean         |  0.213996 |
| TMAX_3d_mean         |  0.207968 |
| TMAX_lag3d           |  0.204402 |
| TMAX_lag2d           |  0.204352 |

### Target: `customers_out`
| feature              |     corr |
|:---------------------|---------:|
| customers_out_mean   | 0.93992  |
| cust_minute_area     | 0.939683 |
| pct_out_area_covered | 0.892544 |
| pct_out_max          | 0.883938 |
| pct_out_max_unified  | 0.883929 |
| pct_out_area_unified | 0.880886 |
| pct_out_area         | 0.880883 |
| damaging_winds       | 0.172872 |
| high_winds           | 0.139237 |
| PRCP_3d_sum          | 0.133372 |
| WSF2_7d_max          | 0.131082 |
| WSF2_3d_max          | 0.128378 |
| PRCP_3d_max          | 0.125104 |
| PRCP                 | 0.123017 |
| precip_volatility_7d | 0.117152 |
| PRCP_7d_sum          | 0.117081 |
| PRCP_7d_max          | 0.115175 |
| rapid_wind_increase  | 0.102947 |
| WSF5                 | 0.100196 |
| strong_winds         | 0.098902 |

## 10) Constant & Near-Zero-Variance
- Constant columns: **0**
- Near-zero-variance (unique_frac < 1%): **60**
|                         |   unique_count |
|:------------------------|---------------:|
| WSF2                    |             55 |
| WSF5                    |             83 |
| WT01                    |              2 |
| WT02                    |              2 |
| WT03                    |              2 |
| WT04                    |              2 |
| WT05                    |              2 |
| WT08                    |              2 |
| WT06                    |              2 |
| WT11                    |              2 |
| WSF2_lag1d              |             55 |
| WSF2_lag2d              |             55 |
| WSF2_3d_max             |             53 |
| WSF2_7d_max             |             52 |
| heavy_rain              |              2 |
| extreme_rain            |              2 |
| heat_wave               |              2 |
| extreme_heat            |              2 |
| freezing                |              2 |
| extreme_cold            |              2 |
| high_winds              |              2 |
| damaging_winds          |              2 |
| consecutive_heat_days   |             68 |
| consecutive_freeze_days |             74 |
| freeze_thaw_cycle       |              2 |
| light_rain              |              2 |
| moderate_rain           |              2 |
| days_since_rain         |              2 |
| wet_period_indicator    |              2 |
| moderate_winds          |              2 |
| strong_winds            |              2 |
| sustained_winds_3d      |              2 |
| rapid_wind_increase     |              2 |
| month                   |             12 |
| season                  |              4 |
| winter_freeze           |              2 |
| summer_heat             |              2 |
| spring_storms           |              2 |
| ice_storm_risk          |              2 |
| wet_windy_combo         |              2 |
| heat_demand_stress      |              2 |
| multiple_extremes       |              2 |
| weather_severity_score  |              8 |
| county_fips             |             12 |
| county_name             |             12 |
| state                   |              9 |
| climate_zone            |              5 |
| outage_risk_profile     |             12 |
| year                    |              6 |
| fips_code               |             12 |

## 11) Skewness & IQR Outliers (numeric only)
Top skewness (first 30):
|                         |     skew |
|:------------------------|---------:|
| days_since_rain         | 56.9781  |
| damaging_winds          | 53.7163  |
| cust_minute_area        | 53.2081  |
| customers_out_mean      | 51.9521  |
| customers_out           | 49.8533  |
| pct_out_area            | 39.923   |
| pct_out_area_covered    | 39.3387  |
| pct_out_area_unified    | 39.1529  |
| extreme_heat            | 34.3313  |
| pct_out_max             | 31.9316  |
| pct_out_max_unified     | 31.2833  |
| spring_storms           | 28.9064  |
| extreme_cold            | 14.0347  |
| strong_winds            | 13.8205  |
| high_winds              | 13.3714  |
| rapid_wind_increase     | 12.2443  |
| consecutive_heat_days   | 12.2055  |
| consecutive_freeze_days |  9.55971 |
| PRCP                    |  7.36718 |
| PRCP_lag3d              |  7.36668 |
| PRCP_lag2d              |  7.36659 |
| PRCP_lag1d              |  7.36598 |
| AWND                    |  6.44666 |
| consecutive_rain_days   |  6.12806 |
| summer_heat             |  5.24496 |
| PRCP_3d_sum             |  5.09393 |
| PRCP_3d_max             |  4.96349 |
| wet_windy_combo         |  4.92532 |
| sustained_winds_3d      |  4.60464 |
| heat_wave               |  4.56135 |

Outlier counts by IQR rule (top 50):
|                         |   outlier_count |
|:------------------------|----------------:|
| PRCP_lag2d              |            4341 |
| PRCP                    |            4339 |
| PRCP_lag1d              |            4337 |
| PRCP_lag3d              |            4335 |
| heating_degree_days     |            3586 |
| PRCP_3d_sum             |            3428 |
| PRCP_3d_max             |            3272 |
| consecutive_rain_days   |            3075 |
| pct_out_max             |            2498 |
| pct_out_max_unified     |            2403 |
| pct_out_area            |            2355 |
| PRCP_7d_sum             |            2331 |
| pct_out_area_covered    |            2312 |
| PRCP_7d_max             |            2284 |
| pct_out_area_unified    |            2269 |
| cust_minute_area        |            2227 |
| minutes_out             |            2213 |
| precip_volatility_7d    |            2205 |
| customers_out_mean      |            2169 |
| customers_out           |            2062 |
| coverage                |            1925 |
| snapshots_count         |            1925 |
| minutes_observed        |            1925 |
| PRCP_14d_sum            |            1761 |
| mechanical_stress_index |            1452 |
| temp_change_1d          |            1277 |
| wind_acceleration_1d    |            1136 |
| temp_volatility_3d      |             972 |
| WSF2                    |             643 |
| WSF2_lag1d              |             643 |
| WSF2_lag2d              |             643 |
| WSF2_3d_max             |             641 |
| WSF5                    |             575 |
| WSF2_7d_max             |             572 |
| AWND                    |             532 |
| num_out_per_day         |             294 |
| TMAX_14d_max            |             150 |
| temp_range_daily        |             108 |
| WSF2_7d_mean            |             108 |
| thermal_stress_index    |              95 |
| TMIN_14d_min            |              93 |
| TMIN_7d_min             |              66 |
| TMIN_lag2d              |              55 |
| TMIN_lag1d              |              55 |
| TMIN                    |              55 |
| TMIN_lag3d              |              55 |
| TMAX_lag2d              |              41 |
| TMAX                    |              41 |
| TMAX_lag1d              |              41 |
| TMAX_lag3d              |              41 |

## 12) Memory Usage
- Total: **32.24 MB**
|                     |            bytes |
|:--------------------|-----------------:|
| outage_risk_profile |      1.80854e+06 |
| climate_zone        |      1.6576e+06  |
| county_name         |      1.63913e+06 |
| county_fips         |      1.53359e+06 |
| run_start_time_day  |      1.5067e+06  |
| season              |      1.41667e+06 |
| fips_code           |      1.40362e+06 |
| state               |      1.32564e+06 |
| train_mask          | 931764           |
| day                 | 207944           |
| PRCP                | 207944           |
| AWND                | 207944           |
| WT08                | 207944           |
| WT06                | 207944           |
| WT11                | 207944           |
| PRCP_lag1d          | 207944           |
| PRCP_lag2d          | 207944           |
| PRCP_lag3d          | 207944           |
| TMAX_lag1d          | 207944           |
| TMAX_lag2d          | 207944           |
| TMAX_lag3d          | 207944           |
| TMIN_lag1d          | 207944           |
| TMIN_lag2d          | 207944           |
| TMIN_lag3d          | 207944           |
| WSF2_lag1d          | 207944           |
| WSF2_lag2d          | 207944           |
| PRCP_3d_sum         | 207944           |
| PRCP_7d_sum         | 207944           |
| PRCP_14d_sum        | 207944           |
| PRCP_3d_max         | 207944           |
| PRCP_7d_max         | 207944           |
| TMAX_3d_mean        | 207944           |
| TMAX_7d_mean        | 207944           |
| TMAX_7d_max         | 207944           |
| TMAX_14d_max        | 207944           |
| TMAX                | 207944           |
| TMIN                | 207944           |
| WSF2                | 207944           |
| WSF5                | 207944           |
| WT01                | 207944           |
| WT02                | 207944           |
| WT03                | 207944           |
| WT04                | 207944           |
| WT05                | 207944           |
| extreme_rain        | 207944           |
| heavy_rain          | 207944           |
| WSF2_7d_mean        | 207944           |
| WSF2_7d_max         | 207944           |
| WSF2_3d_max         | 207944           |
| TMIN_14d_min        | 207944           |

## 13) Per-County, Per-Month Outage Rate
|   fips_code | _month   |   rate |   n |
|------------:|:---------|-------:|----:|
|       06073 | 2019-01  |      1 |  31 |
|       06073 | 2019-02  |      1 |  20 |
|       06073 | 2019-04  |      1 |  30 |
|       06073 | 2019-05  |      1 |  31 |
|       06073 | 2019-06  |      1 |   9 |
|       06073 | 2019-07  |      1 |  31 |
|       06073 | 2019-08  |      1 |  31 |
|       06073 | 2019-09  |      1 |  30 |
|       06073 | 2019-10  |      1 |  31 |
|       06073 | 2019-11  |      1 |  21 |
|       06073 | 2020-01  |      1 |  31 |
|       06073 | 2020-02  |      1 |  29 |
|       06073 | 2020-03  |      1 |  14 |
|       06073 | 2020-04  |      1 |  30 |
|       06073 | 2020-05  |      1 |  31 |
|       06073 | 2020-06  |      1 |  16 |
|       06073 | 2020-07  |      1 |  31 |
|       06073 | 2020-08  |      1 |  31 |
|       06073 | 2020-09  |      1 |  10 |
|       06073 | 2020-10  |      1 |  31 |

## 14) Outage Severity Threshold Sweep (pct_out_max)
|   pct_thr |   prevalence |     n |
|----------:|-------------:|------:|
|     0.001 |    0.413227  | 25993 |
|     0.005 |    0.0962182 | 25993 |
|     0.01  |    0.0380872 | 25993 |
|     0.02  |    0.0148502 | 25993 |

## 15) Temporal Cross-Correlation (±2 days)
| feature     | target      |   lag |     corr |
|:------------|:------------|------:|---------:|
| PRCP_7d_sum | minutes_out |    -1 | 0.243614 |
| PRCP_7d_sum | minutes_out |     0 | 0.242347 |
| PRCP_7d_sum | minutes_out |    -2 | 0.242037 |
| PRCP_7d_sum | minutes_out |     1 | 0.233832 |
| PRCP_7d_sum | minutes_out |     2 | 0.22861  |
| TMAX        | minutes_out |     2 | 0.204335 |
| TMAX        | minutes_out |     1 | 0.203552 |
| TMAX        | minutes_out |     0 | 0.202067 |
| TMAX        | minutes_out |    -2 | 0.201205 |
| TMAX        | minutes_out |    -1 | 0.200718 |
| WSF2        | minutes_out |     0 | 0.190589 |
| WSF2        | minutes_out |     1 | 0.189217 |
| WSF2_7d_max | minutes_out |     0 | 0.185488 |
| WSF2_7d_max | minutes_out |    -2 | 0.185272 |
| WSF2_7d_max | minutes_out |    -1 | 0.185053 |
| WSF2_7d_max | minutes_out |     1 | 0.175882 |
| WSF2        | minutes_out |     2 | 0.166278 |
| WSF2_7d_max | minutes_out |     2 | 0.164204 |
| WSF2        | minutes_out |    -1 | 0.15978  |
| PRCP        | minutes_out |     0 | 0.154497 |

## 16) Integrity / Consistency Checks
|                                  |   count |
|:---------------------------------|--------:|
| customers_out_leq_total_viol     |       0 |
| pct_out_max_outside_0_1          |       0 |
| pct_out_area_outside_0_1         |       0 |
| pct_out_area_unified_outside_0_1 |       0 |
| pct_out_area_covered_outside_0_1 |       0 |
| minutes_out_mismatch_fullcov     |       0 |

## 17) Identical Numeric Columns
None

## 18) Year-over-Year Means (selected)
|   _year |     PRCP |    TMAX |   minutes_out |   pct_out_max |   customers_out |
|--------:|---------:|--------:|--------------:|--------------:|----------------:|
|    2019 |  95.6103 | 21.3792 |       1023.67 |    0.00191135 |         1471.67 |
|    2020 | 110.303  | 21.8276 |       1072.73 |    0.00229966 |         1878.51 |
|    2021 | 103.311  | 21.5669 |       1178.27 |    0.00257554 |         2206.99 |
|    2022 |  87.3001 | 21.6954 |       1180.34 |    0.0024283  |         1691.04 |
|    2023 |  95.1193 | 21.907  |       1167.24 |    0.00225186 |         1618.23 |
|    2024 |  96.9241 | 21.969  |       1178.44 |    0.00406368 |         4380.15 |

## 19) Flag Hierarchy Checks
|                            |   count |
|:---------------------------|--------:|
| extreme_implies_heavy_viol |       0 |

## 20) Missingness by County × Month (long)
|   fips_code | day     |   missing_frac | column   |
|------------:|:--------|---------------:|:---------|
|       06073 | 2019-01 |      0         | AWND     |
|       06073 | 2019-02 |      0.05      | AWND     |
|       06073 | 2019-04 |      0         | AWND     |
|       06073 | 2019-05 |      0         | AWND     |
|       06073 | 2019-06 |      0         | AWND     |
|       06073 | 2019-07 |      0         | AWND     |
|       06073 | 2019-08 |      0         | AWND     |
|       06073 | 2019-09 |      0         | AWND     |
|       06073 | 2019-10 |      0         | AWND     |
|       06073 | 2019-11 |      0.047619  | AWND     |
|       06073 | 2020-01 |      0         | AWND     |
|       06073 | 2020-02 |      0         | AWND     |
|       06073 | 2020-03 |      0.0714286 | AWND     |
|       06073 | 2020-04 |      0         | AWND     |
|       06073 | 2020-05 |      0         | AWND     |
|       06073 | 2020-06 |      0.0625    | AWND     |
|       06073 | 2020-07 |      0         | AWND     |
|       06073 | 2020-08 |      0         | AWND     |
|       06073 | 2020-09 |      0         | AWND     |
|       06073 | 2020-10 |      0         | AWND     |

## 21) Fold Planning Summary (year × county)
|   _year |   fips_code |   n |   any_out_rate |
|--------:|------------:|----:|---------------:|
|    2019 |       06073 | 265 |       1        |
|    2019 |       06075 | 365 |       0.99726  |
|    2019 |       12086 | 365 |       0.99726  |
|    2019 |       12095 | 365 |       0.99726  |
|    2019 |       13121 | 365 |       0.99726  |
|    2019 |       17031 | 365 |       0.99726  |
|    2019 |       25025 | 365 |       0.808219 |
|    2019 |       36061 | 365 |       0.282192 |
|    2019 |       48029 | 365 |       0.939726 |
|    2019 |       48201 | 365 |       0.99726  |
|    2019 |       51059 | 365 |       0        |
|    2019 |       53033 | 365 |       0.99726  |
|    2020 |       06073 | 315 |       0.996825 |
|    2020 |       06075 | 366 |       0.997268 |
|    2020 |       12086 | 366 |       0.997268 |
|    2020 |       12095 | 366 |       0.997268 |
|    2020 |       13121 | 366 |       0.997268 |
|    2020 |       17031 | 366 |       0.997268 |
|    2020 |       25025 | 366 |       0.989071 |
|    2020 |       36061 | 366 |       0.961749 |
|    2020 |       48029 | 366 |       0.997268 |
|    2020 |       48201 | 366 |       0.997268 |
|    2020 |       51059 | 366 |       0.136612 |
|    2020 |       53033 | 366 |       0.997268 |
|    2021 |       06073 | 329 |       0.99696  |
|    2021 |       06075 | 365 |       0.961644 |
|    2021 |       12086 | 365 |       0.99726  |
|    2021 |       12095 | 365 |       0.99726  |
|    2021 |       13121 | 365 |       0.99726  |
|    2021 |       17031 | 365 |       0.99726  |
