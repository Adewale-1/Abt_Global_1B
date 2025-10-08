# Outage Daily Labels — Column Guide (2019–2024)

Below is a plain-English description of every column in your combined+labels CSV, plus how each value is calculated. Symbols used below:
- `min` = minutes  
- `cust` = customers  
- `%` results are in **[0, 1]** unless noted

---

### `fips_code`
- **What:** 5-digit county identifier as text (leading zeros preserved).
- **How:** Carried through from source; zero-padded to 5 characters.

### `run_start_time_day`
- **What:** Local calendar day for that county (yyyy-mm-dd).
- **How:** Source timestamps (UTC) → converted to county’s local timezone (Pacific/Central/Eastern) → take the local **civil day**.

### `any_out`
- **What:** Indicator (0/1) that **any** outage snapshot occurred that day.
- **How:** `1` if **any** positive `customers_out` snapshot exists that day for that FIPS; else `0`.

### `num_out_per_day`
- **What:** Number of distinct outage “events” that day.
- **How:** On **positive** snapshots only, sort by time; start a new event if the gap to the previous snapshot is `> 30 min` **or** the local day changed. Count unique events within the day.

### `minutes_out`
- **What:** Total minutes with **outage present** that day.
- **How:** `snapshot_minutes × (# positive snapshots that day)`.
  > Note: This is **not** the length of the reporting window, it’s the sum of minutes when customers were out.

### `customers_out`
- **What:** Peak customers out that day.
- **How:** `max(customers_out)` over that day’s snapshots.

### `customers_out_mean`
- **What:** Average customers out across that day’s snapshots.
- **How:** Arithmetic mean of `customers_out` over the day’s available snapshots.  
  > With event-only/sparse feeds, this is essentially the mean **while out**.

### `cust_minute_area`
- **What:** “Area under the curve” — total customer-minutes.
- **How:** `Σ (customers_out × snapshot_minutes)` across the day. Units: **customer-minutes**.

### `pct_out_max`
- **What:** Peak **fraction** of customers out that day.
- **How:** `customers_out / customers_total`. NaN if `customers_total` is missing/zero.

### `pct_out_area`
- **What:** Legacy percentage-of-day metric (if present from earlier runs). Prefer the unified version below.
- **How:** Same idea as `pct_out_area_unified`, but earlier versions could have assumed 1440 min/day.

### `customers_total`
- **What:** Total customers in the county (denominator).
- **How:** Joined from `MCC.csv` (or from raw if provided). Used to normalize into rates.

### `minutes_in_local_day`
- **What:** Minutes in the local civil day for that FIPS on that date.
- **How:** DST-aware length of the day (can be **1380**, **1440**, or **1500** minutes).

### `snapshot_minutes`
- **What:** Snapshot cadence in minutes.
- **How:** Inferred from the data (snapped to one of **{5, 10, 15, 30, 60}**), or forced via script arg.

### `snapshots_count`
- **What:** Number of **positive** snapshots that day.
- **How:** `minutes_out / snapshot_minutes`.

### `minutes_observed`
- **What:** Total minutes represented by the **positive** snapshots.
- **How:** `snapshots_count × snapshot_minutes`. With event-only input, this equals `minutes_out`.

### `coverage`
- **What:** Fraction of the local day covered by **positive** snapshots.
- **How:** `minutes_observed / minutes_in_local_day`, clipped to [0, 1].  
  > With event-only input, this reads as “fraction of the day with outages recorded,” **not** general feed coverage.

### `pct_out_area_unified`  
- **What:** Average **fraction of customers out over the entire local day**.
- **How:**
  ```text
  pct_out_area_unified = cust_minute_area
                         / (customers_total × minutes_in_local_day)
  ```
  Uses DST-aware day length, so it’s consistent across years and timezones.

### `pct_out_area_covered`  
- **What:** Average **fraction of customers out during minutes when outages were recorded** (i.e., intensity while out).
- **How:**
  ```text
  pct_out_area_covered = cust_minute_area
                         / (customers_total × minutes_observed)
  ```
  Defined when `minutes_observed > 0` and `customers_total > 0`.

### `pct_out_max_unified`
- **What:** Unified (recomputed) peak fraction of customers out.
- **How:** `customers_out / customers_total` with consistent typing and guards.

### `train_mask`
- **What:** Convenience flag for training-quality rows.
- **How:** `True` when `customers_total > 0` **and** `coverage ≥ 0.8`.  


---

## Which column should be our label?

**Primary training label:** **`pct_out_area_covered`**   
- **Why:** It measures the **rate of customers affected** **conditional on observed outage minutes**, i.e., the **intensity while out**. That’s robust when some days have short/fragmented outage windows, avoids diluting intensity just because the local day is long (DST) or because outages only occurred briefly.

**Secondary (reporting / comparability):** **`pct_out_area_unified`**  
- DST-aware full-day rate that’s ideal for **comparing counties/days** on a common basis (entire local day).

**Optional auxiliary targets:**  
- **`pct_out_max_unified`** (peak severity)  
- **Binary:** `any_out` (as a first-stage classifier in a hurdle model, given class imbalance)
