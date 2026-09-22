# Review — Section 1: Foundation and data

## Section

This is Section 1 of `PLAN.md` (steps 1.1 to 1.7): the package skeleton and `Config`, the `FredClient` raw-write conventions, the FRED market pulls and `month_end_market`, the ALFRED vintage pulls and `build_asof`, the French pull, the project 1 adapter, and the as-of panel `data/processed/asof_panel.parquet`. **Every step completed.** The section ran across three sittings because `FRED_API_KEY` was not available at first: 1.1 and 1.2 on 17 September (stopped before 1.3 as instructed), 1.5 and 1.6 on 18 September (the two steps that do not need FRED, built out of numeric order under rule 1's principle of finishing everything that does not depend on the blocker), and 1.3, 1.4 and 1.7 on 22 September once the key was supplied. The key was placed in the User environment scope; it is not in the repo and not in any committed file. Two intermediate review files were committed at the stops (`7cbe025`, `24322a6`); this file supersedes them.

`python -m regime.run --section 1` rebuilds the three as-of tables and the panel from the pinned pull ids in about 20 seconds and is idempotent (the panel's SHA-256 is identical across two consecutive runs, shown below).

## Steps completed

- `step 1.1: package skeleton, Config with load_config, section registry and the key round-trip test` — `ff81cbf`
- `step 1.2: FredClient with one pull_id per session, write_raw that never overwrites, and the append-only manifest` — `1fb67ac`
- `step 1.5: French pull, parse, checksum; pin french.pull_id 20260918T083009Z and sample.end 2026-07-31` — `ff9e132`
- `step 1.6: project 1 adapter that logs project1: absent and returns the empty (date, factor, ret) schema` — `61ac025`
- `step 1.3: market series pulls and month_end_market with the 10-day lookback; pin fred.market_pull_id 20260922T004835Z` — `89b9992`
- `step 1.4: ALFRED vintage pulls and build_asof with the as-of-t release rule; pin fred.vintage_pull_id 20260922T005041Z` — `c468292`
- `step 1.7: build_asof_panel with the sixteen-column schema, planted-future-value and same-vintage lag tests` — `327b8db`

(Listed in commit order; 1.5 and 1.6 precede 1.3 in history for the reason above.)

## Evidence

### Step 1.1 — `Config` has 86 fields, one per `config.toml` key, and every field is `int`, `float`, `str`, `bool` or `tuple`

```
$ .venv/Scripts/python.exe -c "import dataclasses; from regime.config import Config, load_config; cfg = load_config(); [print(f'{f.name:40s} {type(getattr(cfg, f.name)).__name__}') for f in dataclasses.fields(Config)]; print(len(dataclasses.fields(Config)), 'fields')"
run_seed                                 int
run_step_timeout_minutes                 int
run_long_step_timeout_minutes            int
run_long_steps                           tuple
sample_start                             str
sample_end                               str
sample_features_from                     str
sample_first_window_end                  str
sample_robustness_from                   str
fred_api_key_env                         str
fred_market_series                       tuple
fred_vintage_series                      tuple
fred_lookback_days                       int
fred_raw_dir                             str
fred_manifest                            str
fred_market_pull_id                      str
fred_vintage_pull_id                     str
french_factors_url                       str
french_momentum_url                      str
french_columns                           tuple
french_pull_id                           str
features_core                            tuple
features_robustness                      tuple
features_change_lag                      int
features_cpi_lag                         int
features_dollar_splice_date              str
features_ddof                            int
hmm_covariance_type                      str
hmm_n_iter                               int
hmm_tol                                  float
hmm_init_params                          str
hmm_params                               str
hmm_n_restarts                           int
hmm_k_grid                               tuple
hmm_k_candidates                         tuple
hmm_refit_month                          int
hmm_min_state_rows                       int
hmm_assigned_threshold                   float
hmm_anchor_feature                       str
hmm_anchor_tiebreak_feature              str
hmm_anchor_tie_tolerance                 float
hmm_robustness_first_refit               str
hmm_robustness_covariance_type           str
gmm_covariance_type                      str
gmm_n_init                               int
gmm_max_iter                             int
gmm_tol                                  float
gmm_n_restarts                           int
rules_growth_feature                     str
rules_inflation_feature                  str
bootstrap_block_size                     int
bootstrap_n_replications                 int
bootstrap_p_low                          float
bootstrap_p_high                         float
bootstrap_block_size_grid                tuple
strategy_factors                         tuple
strategy_universe                        tuple
strategy_min_regime_obs                  int
strategy_min_regime_obs_grid             tuple
strategy_eta_grid                        tuple
strategy_lag_grid                        tuple
strategy_cost_bp_grid                    tuple
strategy_headline_eta                    float
strategy_headline_lag                    int
strategy_headline_cost_bp                int
strategy_headline_source                 str
strategy_label_sources                   tuple
outputs_interim_dir                      str
outputs_processed_dir                    str
outputs_external_dir                     str
outputs_tables_dir                       str
outputs_charts_dir                       str
outputs_regimes_dir                      str
outputs_chart_dpi                        int
outputs_asof_panel                       str
outputs_features_raw                     str
outputs_features_z                       str
outputs_dropped_rows                     str
outputs_primary_k                        str
outputs_project1_file                    str
outputs_filtered_probs                   str
outputs_smoothed_probs                   str
outputs_gmm_filtered_probs               str
outputs_regime_labels                    str
outputs_timing_results                   str
outputs_runtime_log                      str
86 fields
```

Every `config.toml` array is a `tuple`; `hmm_tol`, `gmm_tol`, `hmm_assigned_threshold`, `hmm_anchor_tie_tolerance`, `bootstrap_p_low`, `bootstrap_p_high` and `strategy_headline_eta` are `float`. At step 1.1 the three `*_pull_id` fields and `sample_end` were the empty `str` placeholders; each was filled by its owning step (1.3, 1.4, 1.5) and `tests/test_config.py` tightened to a format assertion for that key in the same commit.

### Step 1.1 — `load_config` raises `KeyError` naming the offending key in both directions

Checked by hand (not one of the named tests; the named round-trip test covers the committed file): a copy of `config.toml` with an extra `[extra]` table holding `foo = 1`, and a copy with `hmm.n_iter` deleted.

```
extra key -> "config.toml key 'extra_foo' has no Config field"
missing key -> "Config field 'hmm_n_iter' has no config.toml key"
```

### Step 1.1 — `python -m regime.run --section 2` raises `NotImplementedError("section 2 not built")`

```
$ .venv/Scripts/python.exe -m regime.run --section 2
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "C:\Utkarsh\10. Quant Projects\2) Macro Regime Classifier\regime\run.py", line 65, in <module>
    main()
  File "C:\Utkarsh\10. Quant Projects\2) Macro Regime Classifier\regime\run.py", line 61, in main
    SECTIONS[n](cfg, pull=args.pull if n == 1 else False)
  File "C:\Utkarsh\10. Quant Projects\2) Macro Regime Classifier\regime\run.py", line 26, in section
    raise NotImplementedError(f"section {n} not built")
NotImplementedError: section 2 not built
```

`SECTIONS` has keys 1 … 7 in order (`tests/test_run.py::test_sections_registered_in_order`). Sections 2 to 7 still raise `NotImplementedError`; section 1 got its body across steps 1.3, 1.4 and 1.7.

### Step 1.1 — `pytest -q` at the end of the step

```
$ pytest -q
....                                                                     [100%]
4 passed in 0.02s
```

### Step 1.2 — the exception text, the manifest header line and one manifest row

Run with `FRED_API_KEY` removed from the environment, in a temporary directory (config replaced via `dataclasses.replace` to point `fred_raw_dir` and `fred_manifest` there); a 3-row frame with one NaN written once, then written again with the same `(source, series, pull_id)`.

```
$ env -u FRED_API_KEY .venv/Scripts/python.exe -c "<FredClient(cfg); write_raw twice; print the manifest file>"
exception: RuntimeError('FRED_API_KEY not set')
second write: FileExistsError
manifest file:
source,series,pull_id,rows,first_date,last_date,sha256,pulled_at
fred,DGS10,20260917T120000Z,3,2000-01-01,2000-01-03,ce3e63c97cb1145bb9ba6b3d03a3e13bd709d3916b9ec149763d8f7d6cb8468c,2026-09-17T14:15:00+00:00

sha256 of parquet bytes: ce3e63c97cb1145bb9ba6b3d03a3e13bd709d3916b9ec149763d8f7d6cb8468c
parquet path: raw\fred\DGS10_20260917T120000Z.parquet
```

The `sha256` column equals the SHA-256 of the parquet file bytes; the header line is exactly `source,series,pull_id,rows,first_date,last_date,sha256,pulled_at`; the second write raised `FileExistsError` and the manifest still has one data row.

### Step 1.2 — `pytest -q` at the end of the step

```
$ pytest -q
.........                                                                [100%]
9 passed in 0.82s
```

### Step 1.3 — one client, seven series, one `pull_id`; manifest rows; `head(3)`/`tail(3)` of each month-end series; the NaN table

`python -m regime.run --section 1 --pull` on 22 September (log below) pulled all seven `fred.market_series` through one `FredClient`; every line carries the same `pull_id 20260922T004835Z`, pinned as `fred.market_pull_id` in the same commit.

```
$ .venv/Scripts/python.exe -m regime.run --section 1 --pull   (step 1.3 state of run.py; market part)
INFO regime: pulled market series DGS10 under pull_id 20260922T004835Z
INFO regime: pulled market series DGS2 under pull_id 20260922T004835Z
INFO regime: pulled market series DTWEXBGS under pull_id 20260922T004835Z
INFO regime: pulled market series DTWEXM under pull_id 20260922T004835Z
INFO regime: pulled market series DCOILWTICO under pull_id 20260922T004835Z
INFO regime: pulled market series VIXCLS under pull_id 20260922T004835Z
INFO regime: pulled market series T10YIE under pull_id 20260922T004835Z
INFO regime: market pull_id 20260922T004835Z (pin as fred.market_pull_id in config.toml)
INFO regime: fred.market_pull_id not pinned; no month-end series built
```

```
$ python ev_1_3.py   (manifest rows for the pull; month_end_market(series, cfg.fred_market_pull_id, cfg) head/tail; NaN summary)
manifest rows, pull_id 20260922T004835Z
  source      series           pull_id   rows  first_date   last_date                                                            sha256                         pulled_at
3   fred       DGS10  20260922T004835Z  16884  1962-01-02  2026-09-18  3bffe315b62b43fbf2b9de074ec4974a52d31a9b70100a226fa001b983384b9f  2026-09-22T00:48:35.858507+00:00
4   fred        DGS2  20260922T004835Z  13124  1976-06-01  2026-09-18  bc42d4c0b4c50531926266fac9b6cb672395a7214434067d0597b92eb1c6aa73  2026-09-22T00:48:35.858507+00:00
5   fred    DTWEXBGS  20260922T004835Z   5405  2006-01-02  2026-09-18  8d36f9816a29bf473458a6429209b4a9af182bf9cf8eda986e606b91c3336429  2026-09-22T00:48:35.858507+00:00
6   fred      DTWEXM  20260922T004835Z  12261  1973-01-02  2019-12-31  95d20c4f20a751c59fab0c8fc23b0f484850065806175895cfb4cde4dc3e454e  2026-09-22T00:48:35.858507+00:00
7   fred  DCOILWTICO  20260922T004835Z  10619  1986-01-02  2026-09-15  8a603915f8d28cdff8dac536f009e18cc4bab8fe9d5445a5d0c53ed7fcf3ac89  2026-09-22T00:48:35.858507+00:00
8   fred      VIXCLS  20260922T004835Z   9579  1990-01-02  2026-09-18  b9b2aa504dd2a8df13cbd3e51d65719750db82f10dc331c79fd7c9529b49cf60  2026-09-22T00:48:35.858507+00:00
9   fred      T10YIE  20260922T004835Z   6188  2003-01-02  2026-09-21  1b9e1f51bad8e4f25f7815bfc82e9af789d724da15df93b16f144dc952625184  2026-09-22T00:48:35.858507+00:00

DGS10: head(3)
date
1990-01-31    8.43
1990-02-28    8.51
1990-03-31    8.65
Freq: ME
DGS10: tail(3)
date
2026-06-30    4.44
2026-07-31    4.75
2026-08-31    4.75
Freq: ME

DGS2: head(3)
date
1990-01-31    8.28
1990-02-28    8.43
1990-03-31    8.64
Freq: ME
DGS2: tail(3)
date
2026-06-30    4.14
2026-07-31    4.28
2026-08-31    4.34
Freq: ME

DTWEXBGS: head(3)
date
1990-01-31   NaN
1990-02-28   NaN
1990-03-31   NaN
Freq: ME
DTWEXBGS: tail(3)
date
2026-06-30    120.9248
2026-07-31    119.7034
2026-08-31    118.5679
Freq: ME

DTWEXM: head(3)
date
1990-01-31    92.3441
1990-02-28    93.2150
1990-03-31    94.4002
Freq: ME
DTWEXM: tail(3)
date
2019-10-31    91.7069
2019-11-30    92.7133
2019-12-31    90.8221
Freq: ME

DCOILWTICO: head(3)
date
1990-01-31    22.69
1990-02-28    21.55
1990-03-31    20.34
Freq: ME
DCOILWTICO: tail(3)
date
2026-06-30    70.56
2026-07-31    86.16
2026-08-31    87.03
Freq: ME

VIXCLS: head(3)
date
1990-01-31    25.36
1990-02-28    21.99
1990-03-31    19.73
Freq: ME
VIXCLS: tail(3)
date
2026-06-30    16.45
2026-07-31    15.99
2026-08-31    14.92
Freq: ME

T10YIE: head(3)
date
1990-01-31   NaN
1990-02-28   NaN
1990-03-31   NaN
Freq: ME
T10YIE: tail(3)
date
2026-06-30    2.24
2026-07-31    2.28
2026-08-31    2.31
Freq: ME

NaN summary
    series  n_months      first       last  n_nan                                       nan_dates
     DGS10       440 1990-01-31 2026-08-31      0                                               -
      DGS2       440 1990-01-31 2026-08-31      0                                               -
  DTWEXBGS       440 1990-01-31 2026-08-31    192 1990-01 .. 2005-12 (192 months, contiguous=yes)
    DTWEXM       360 1990-01-31 2019-12-31      0                                               -
DCOILWTICO       440 1990-01-31 2026-08-31      0                                               -
    VIXCLS       440 1990-01-31 2026-08-31      0                                               -
    T10YIE       440 1990-01-31 2026-08-31    156 1990-01 .. 2002-12 (156 months, contiguous=yes)
```

NaN dates against the plan's expectation: DTWEXBGS is NaN for the 192 month-ends 1990-01 to 2005-12 (its raw history starts 2006-01-02); T10YIE for the 156 month-ends 1990-01 to 2002-12 (raw history starts 2003-01-02); DTWEXM's raw history ends 2019-12-31, so `month_end_market` stops at 2019-12-31 (360 month-ends, none NaN) and the panel below shows it NaN from 2020-01 onward (79 rows) after reindexing; VIXCLS's raw history starts 1990-01-02, so 1990-01-31 has a value and the series has no NaN. DGS10, DGS2 and DCOILWTICO have no NaN month-end in 1990-01 to 2026-08 under the 10-day lookback.

### Step 1.3 — `pytest -q` at the end of the step

```
$ pytest -q
.....................                                                    [100%]
21 passed in 0.88s
```

### Step 1.4 — one client, three vintage series; the pre-1995 check; distinct `realtime_start` counts; the 100,000-row cap

`fredapi.get_series_all_releases` sends no `limit`, so FRED's default cap of 100,000 rows would truncate silently at the latest observation months rather than the earliest release dates; the row counts were checked against the cap as well as running the plan's pre-1995 check. All three pass both, so no `OPEN.md` item and the as-of tables were built.

```
$ (one FredClient; pull_vintages for each of cfg.fred_vintage_series; load_releases and report)
CPIAUCSL: pull_id 20260922T005041Z rows 3362 distinct_realtime_start 669 earliest_realtime_start 1972-07-21 latest_realtime_start 2026-09-11 first_obs 1947-01-01 last_obs 2026-08-01 pre1995 True hit_100000_cap False
INDPRO: pull_id 20260922T005041Z rows 39362 distinct_realtime_start 1223 earliest_realtime_start 1927-01-26 latest_realtime_start 2026-09-18 first_obs 1919-01-01 last_obs 2026-08-01 pre1995 True hit_100000_cap False
UNRATE: pull_id 20260922T005041Z rows 2198 distinct_realtime_start 799 earliest_realtime_start 1960-03-15 latest_realtime_start 2026-09-04 first_obs 1948-01-01 last_obs 2026-08-01 pre1995 True hit_100000_cap False
vintage pull_id 20260922T005041Z
```

### Step 1.4 — manifest rows; per series the release counts and earliest `realtime_start`; `build_asof` rows at 1995-01-31, 2008-12-31 and the last decision date; decision dates with no vintage; the t − 1 statistic with every failing date

```
$ .venv/Scripts/python.exe -m regime.run --section 1   (builds data/interim/asof_<series>_<pull_id>.parquet from the pinned pull)
INFO regime: DGS10: 440 month-ends, 0 NaN
INFO regime: DGS2: 440 month-ends, 0 NaN
INFO regime: DTWEXBGS: 440 month-ends, 192 NaN
INFO regime: DTWEXM: 360 month-ends, 0 NaN
INFO regime: DCOILWTICO: 440 month-ends, 0 NaN
INFO regime: VIXCLS: 440 month-ends, 0 NaN
INFO regime: T10YIE: 440 month-ends, 156 NaN
INFO regime: CPIAUCSL: as-of table 324060 rows, 440 decision dates
INFO regime: INDPRO: as-of table 471900 rows, 440 decision dates
INFO regime: UNRATE: as-of table 318780 rows, 440 decision dates
```

```
$ python ev_1_4.py
manifest rows, pull_id 20260922T005041Z
    source    series           pull_id   rows  first_date   last_date                                                            sha256                         pulled_at
10  alfred  CPIAUCSL  20260922T005041Z   3362  1947-01-01  2026-08-01  2f04bd59933cb1df0fe06cc414d9759dc49ca564700d6f82770b41bf8c0d3d53  2026-09-22T00:50:41.552786+00:00
11  alfred    INDPRO  20260922T005041Z  39362  1919-01-01  2026-08-01  0b820268370f46dbafd1d3a35deea39c852345e796b89a26867bdf69981ff7e9  2026-09-22T00:50:41.552786+00:00
12  alfred    UNRATE  20260922T005041Z   2198  1948-01-01  2026-08-01  f02e8ece72a49e4501a89da6e87c5922abf9fa537e9a2e888330e6b9fed68ec2  2026-09-22T00:50:41.552786+00:00

=== CPIAUCSL: rows 3362, distinct realtime_start 669, earliest 1972-07-21, latest 2026-09-11, first obs 1947-01-01, last obs 2026-08-01
first 3 raw release rows (earliest realtime_start):
      date realtime_start  value
1970-12-01     1972-07-21 119.03
1971-01-01     1972-07-21 119.36
1971-02-01     1972-07-21 119.65
as-of table: 324060 rows, 440 decision dates 1990-01-31 .. 2026-08-31
rows at decision date 1995-01-31, last three observation months with a value:
decision_date  obs_month  value
   1995-01-31 1994-10-31  149.5
   1995-01-31 1994-11-30  149.9
   1995-01-31 1994-12-31  150.2
rows at decision date 2008-12-31, last three observation months with a value:
decision_date  obs_month   value
   2008-12-31 2008-09-30 218.813
   2008-12-31 2008-10-31 216.710
   2008-12-31 2008-11-30 213.060
rows at decision date 2026-08-31, last three observation months with a value:
decision_date  obs_month   value
   2026-08-31 2026-05-31 333.979
   2026-08-31 2026-06-30 332.568
   2026-08-31 2026-07-31 332.813
decision dates with no available vintage (all values NaN): 0 
latest obs month == t-1 for 438 of 440 decision dates = 99.55%
value counts of (t - latest obs month) in months:
1    438
2      2
failing dates:
decision_date latest_obs_month   expected
   1996-01-31       1995-11-30 1995-12-31
   2025-11-30       2025-09-30 2025-10-31

=== INDPRO: rows 39362, distinct realtime_start 1223, earliest 1927-01-26, latest 2026-09-18, first obs 1919-01-01, last obs 2026-08-01
first 3 raw release rows (earliest realtime_start):
      date realtime_start  value
1919-01-01     1927-01-26   83.0
1919-02-01     1927-01-26   80.0
1919-03-01     1927-01-26   77.0
as-of table: 471900 rows, 440 decision dates 1990-01-31 .. 2026-08-31
rows at decision date 1995-01-31, last three observation months with a value:
decision_date  obs_month  value
   1995-01-31 1994-10-31  119.4
   1995-01-31 1994-11-30  120.3
   1995-01-31 1994-12-31  121.4
rows at decision date 2008-12-31, last three observation months with a value:
decision_date  obs_month    value
   2008-12-31 2008-09-30 105.2479
   2008-12-31 2008-10-31 106.7786
   2008-12-31 2008-11-30 106.1173
rows at decision date 2026-08-31, last three observation months with a value:
decision_date  obs_month    value
   2026-08-31 2026-05-31 102.5099
   2026-08-31 2026-06-30 102.7868
   2026-08-31 2026-07-31 102.9939
decision dates with no available vintage (all values NaN): 0 
latest obs month == t-1 for 438 of 440 decision dates = 99.55%
value counts of (t - latest obs month) in months:
1    438
2      1
3      1
failing dates:
decision_date latest_obs_month   expected
   2025-10-31       2025-08-31 2025-09-30
   2025-11-30       2025-08-31 2025-10-31

=== UNRATE: rows 2198, distinct realtime_start 799, earliest 1960-03-15, latest 2026-09-04, first obs 1948-01-01, last obs 2026-08-01
first 3 raw release rows (earliest realtime_start):
      date realtime_start  value
1948-01-01     1960-03-15    3.5
1948-02-01     1960-03-15    3.8
1948-03-01     1960-03-15    4.0
as-of table: 318780 rows, 440 decision dates 1990-01-31 .. 2026-08-31
rows at decision date 1995-01-31, last three observation months with a value:
decision_date  obs_month  value
   1995-01-31 1994-10-31    5.7
   1995-01-31 1994-11-30    5.6
   1995-01-31 1994-12-31    5.4
rows at decision date 2008-12-31, last three observation months with a value:
decision_date  obs_month  value
   2008-12-31 2008-09-30    6.1
   2008-12-31 2008-10-31    6.5
   2008-12-31 2008-11-30    6.7
rows at decision date 2026-08-31, last three observation months with a value:
decision_date  obs_month  value
   2026-08-31 2026-05-31    4.3
   2026-08-31 2026-06-30    4.2
   2026-08-31 2026-07-31    4.1
decision dates with no available vintage (all values NaN): 0 
latest obs month == t-1 for 438 of 440 decision dates = 99.55%
value counts of (t - latest obs month) in months:
1    438
2      2
failing dates:
decision_date latest_obs_month   expected
   2025-10-31       2025-08-31 2025-09-30
   2025-11-30       2025-09-30 2025-10-31
```

The `test_cpi_obs_month_is_t_minus_1` claim: 438 of 440 decision dates (99.55%) have the latest CPI observation month equal to t − 1. The two failing dates are both delayed releases: December 1995 CPI was published late because of the December 1995 to January 1996 federal shutdown (at 1996-01-31 the vintage ends at November 1995), and September 2025 CPI was published late because of the October to November 2025 shutdown (at 2025-11-30 the vintage ends at September 2025). INDPRO and UNRATE show the same 438/440 = 99.55%, with the failures on 2025-10-31 and 2025-11-30 (INDPRO's August 2025 value was the latest at both). No decision date has an empty vintage for any of the three series.

### Step 1.4 — `pytest -q` at the end of the step

```
$ pytest -q
.........................                                                [100%]
25 passed in 0.88s
```

### Step 1.5 — manifest rows for `factors`, `ff5_zip`, `mom_zip`; `head(3)`, the 2010-01-31 row and `tail(3)` of the parsed frame; last month of each file; `diff_french`

Pull `20260918T083009Z` on 18 September 2026 from the two URLs in `config.toml`. The 2010-01 values hard-coded in `tests/test_french.py::test_2010_01_row_matches_site` were read from the downloaded CSVs inside the zips: `F-F_Research_Data_5_Factors_2x3.csv` line `201001,   -3.35,    0.40,    0.33,   -1.08,    0.50,    0.00` and `F-F_Momentum_Factor.csv` line `201001,  -5.31`. Both files say `This file was created using the 202607 CRSP database.`; the last monthly row of each is `202607`, so `sample.end = 2026-07-31`.

```
manifest rows for pull 20260918T083009Z

   source   series           pull_id  rows  first_date   last_date                                                            sha256                         pulled_at

0  french  factors  20260918T083009Z   757  1963-07-31  2026-07-31  1097b15ca824ad8715ce167ec541ce9eac03378b757955c55bff69a1d912e034  2026-09-18T08:30:09.450890+00:00

1  french  ff5_zip  20260918T083009Z   757  1963-07-31  2026-07-31  b8653b411cc5e28917e7ef643bb42f6d2d3703f84bc170eb6ae38d5267c65807  2026-09-18T08:30:09.450890+00:00

2  french  mom_zip  20260918T083009Z  1195  1927-01-31  2026-07-31  7ee14e892b0f7044902fdbc4e25cfaf175b73d4eda0ae6f4a0354a4433afe065  2026-09-18T08:30:09.450890+00:00



parsed factors frame: head(3)

            Mkt-RF     SMB     HML     RMW     CMA     UMD      RF

date                                                              

1963-07-31 -0.0039 -0.0048 -0.0084  0.0064 -0.0115  0.0101  0.0027

1963-08-31  0.0508 -0.0080  0.0172  0.0040 -0.0038  0.0100  0.0025

1963-09-30 -0.0157 -0.0043 -0.0002 -0.0078  0.0015  0.0012  0.0027



row 2010-01-31

            Mkt-RF    SMB     HML     RMW    CMA     UMD   RF

date                                                         

2010-01-31 -0.0335  0.004  0.0033 -0.0108  0.005 -0.0531  0.0



tail(3)

            Mkt-RF     SMB     HML     RMW     CMA     UMD      RF

date                                                              

2026-05-31  0.0491 -0.0265 -0.0231 -0.0816 -0.0146  0.0139  0.0031

2026-06-30 -0.0107  0.0436  0.0358 -0.0507  0.0345  0.0462  0.0029

2026-07-31 -0.0060 -0.0012  0.0211  0.1105  0.0234 -0.1225  0.0033



last month, ff5 file: 2026-07-31 rows 757

last month, momentum file: 2026-07-31 rows 1195

sample_end(pull_id): 2026-07-31 | config sample.end: 2026-07-31

joined rows: 757 first 1963-07-31 last 2026-07-31



diff_french(first, first) - first pull, nothing to compare against; the self-diff is empty:

<empty DataFrame, columns: date, column, first_value, new_value>
```

The momentum file (1195 rows from 1927-01) starts before the 5-factor file (757 rows from 1963-07); the inner join has 757 rows, 1963-07-31 to 2026-07-31. `diff_french` is empty on this, the first pull: there is no earlier snapshot, so the self-diff is shown to demonstrate the empty schema. `run.py --pull` prints `diff_french(new, cfg.french_pull_id)` on any later pull.

### Step 1.5 — `pytest -q` at the end of the step

```
$ pytest -q
............                                                             [100%]
12 passed in 1.41s
```

### Step 1.6 — the `project1: absent` log line and the returned frame's `dtypes`

From the final `python -m regime.run --section 1` (full log under step 1.7): `INFO regime: project1: absent` followed by `INFO regime: project1: 0 rows`. The `dtypes` of the returned frame, from a direct call with the same logger configuration:

```
$ .venv/Scripts/python.exe -c "import logging; logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s'); from regime.data.project1 import load_project1; f = load_project1(); print('shape', f.shape); print(f.dtypes.to_string())"
INFO regime: project1: absent
shape (0, 3)
date      datetime64[ns]
factor            object
ret              float64
```

(`factor` shows as `object` in `dtypes`; it is the string column of an empty pandas 2.2 frame.)

### Step 1.6 — `pytest -q` at the end of the step

```
$ pytest -q
..............                                                           [100%]
14 passed in 0.80s
```

### Step 1.7 — `python -m regime.run --section 1` end to end, and idempotence

```
$ .venv/Scripts/python.exe -m regime.run --section 1
INFO regime: DGS10: 440 month-ends, 0 NaN
INFO regime: DGS2: 440 month-ends, 0 NaN
INFO regime: DTWEXBGS: 440 month-ends, 192 NaN
INFO regime: DTWEXM: 360 month-ends, 0 NaN
INFO regime: DCOILWTICO: 440 month-ends, 0 NaN
INFO regime: VIXCLS: 440 month-ends, 0 NaN
INFO regime: T10YIE: 440 month-ends, 156 NaN
INFO regime: CPIAUCSL: as-of table 324060 rows, 440 decision dates
INFO regime: INDPRO: as-of table 471900 rows, 440 decision dates
INFO regime: UNRATE: as-of table 318780 rows, 440 decision dates
INFO regime: project1: absent
INFO regime: project1: 0 rows
INFO regime: as-of panel written to data/processed/asof_panel.parquet: 439 rows x 16 columns
```

```
$ (sha256 of data/processed/asof_panel.parquet after two consecutive runs)
panel sha256 run1=bac7564215c3d013 run2=bac7564215c3d013 identical=yes
```

### Step 1.7 — `panel.head(3)`, the rows for 2004-12-31 and `sample_end`, the (column, n_nan, first_valid, last_valid) table, NaN decision dates from `features_from`, and `cpi_obs_month − t`

```
$ python ev_1_7.py
panel: 439 rows x 16 columns, index 1990-01-31 .. 2026-07-31, name='date'

head(3)
            dgs10  dgs2  dtwexbgs   dtwexm    wti    vix  t10yie  cpi_m  cpi_m3  indpro_m  indpro_m12  unrate_m  unrate_m12 cpi_obs_month indpro_obs_month unrate_obs_month
date                                                                                                                                                                       
1990-01-31   8.43  8.28       NaN  92.3441  22.69  25.36     NaN  126.3   124.7     142.8       140.4       5.3         5.3    1989-12-31       1989-12-31       1989-12-31
1990-02-28   8.51  8.43       NaN  93.2150  21.55  21.99     NaN  127.7   125.4     140.9       140.8       5.3         5.4    1990-01-31       1990-01-31       1990-01-31
1990-03-31   8.65  8.64       NaN  94.4002  20.34  19.73     NaN  128.3   125.8     141.8       140.5       5.3         5.2    1990-02-28       1990-02-28       1990-02-28

rows for 2004-12-31 and sample.end 2026-07-31
            dgs10  dgs2  dtwexbgs   dtwexm    wti    vix  t10yie    cpi_m   cpi_m3  indpro_m  indpro_m12  unrate_m  unrate_m12 cpi_obs_month indpro_obs_month unrate_obs_month
date                                                                                                                                                                          
2004-12-31   4.24  3.08       NaN  79.4917  43.36  13.29    2.56  191.200  189.400  116.9560    112.6570       5.4         5.9    2004-11-30       2004-11-30       2004-11-30
2026-07-31   4.75  4.28  119.7034      NaN  86.16  15.99    2.28  332.568  330.293  102.6395    101.4785       4.2         4.1    2026-06-30       2026-06-30       2026-06-30

(column, n_nan, first_valid, last_valid)
          column  n_nan first_valid last_valid
           dgs10      0  1990-01-31 2026-07-31
            dgs2      0  1990-01-31 2026-07-31
        dtwexbgs    192  2006-01-31 2026-07-31
          dtwexm     79  1990-01-31 2019-12-31
             wti      0  1990-01-31 2026-07-31
             vix      0  1990-01-31 2026-07-31
          t10yie    156  2003-01-31 2026-07-31
           cpi_m      0  1990-01-31 2026-07-31
          cpi_m3      1  1990-01-31 2026-07-31
        indpro_m      0  1990-01-31 2026-07-31
      indpro_m12      0  1990-01-31 2026-07-31
        unrate_m      0  1990-01-31 2026-07-31
      unrate_m12      0  1990-01-31 2026-07-31
   cpi_obs_month      0  1990-01-31 2026-07-31
indpro_obs_month      0  1990-01-31 2026-07-31
unrate_obs_month      0  1990-01-31 2026-07-31

decision dates from 1991-01-31 with NaN in any of ['dgs10', 'dgs2', 'wti', 'vix', 'cpi_m', 'cpi_m3', 'indpro_m', 'indpro_m12']: 1
            dgs10  dgs2    wti    vix    cpi_m  cpi_m3  indpro_m  indpro_m12
date                                                                        
2026-02-28   3.97  3.38  66.96  19.86  326.588     NaN  102.3412    100.0647

cpi_obs_month - t in months, value counts
-2      2
-1    437
dates where cpi_obs_month - t != -1:
              cpi_m cpi_obs_month
date                             
1996-01-31  153.700    1995-11-30
2025-11-30  324.368    2025-09-30

indpro_obs_month - t in months, value counts
-3      1
-2      1
-1    437

unrate_obs_month - t in months, value counts
-2      2
-1    437
```

The panel has 439 rows (1990-01-31 to 2026-07-31 = `sample.end`), against the 440 month-ends the market series and as-of tables carry to 2026-08-31: the French data is the binding constraint (convention 10). Column NaN counts: `dtwexbgs` 192 (before 2006-01), `dtwexm` 79 (after 2019-12), `t10yie` 156 (before 2003-01), `cpi_m3` 1; every other column has none.

**The single NaN in the eight core columns from `features_from` onward** is `cpi_m3` at 2026-02-28. At that decision date the latest CPI month in the vintage is January 2026 and `cpi_m3` reads October 2025 from the same vintage. October 2025 CPI was never published (the 2025 shutdown); FRED's only release row for that observation month carries a missing value:

```
CPIAUCSL as-of 2026-02-28, observation months from 2025-08:
decision_date  obs_month   value
   2026-02-28 2025-08-31 323.291
   2026-02-28 2025-09-30 324.245
   2026-02-28 2025-10-31     NaN
   2026-02-28 2025-11-30 325.063
   2026-02-28 2025-12-31 326.031
   2026-02-28 2026-01-31 326.588
   2026-02-28 2026-02-28     NaN

raw releases for observation month 2025-10-01:
      date realtime_start  value
2025-10-01     2025-12-18    NaN
```

This is the as-of rule working as specified, not a data-handling fault: the value that did not exist at t is NaN at t. It will drop that one row from `cpi_3m_ann` in section 2, which is the intended treatment (`dropped_rows.csv`).

`cpi_obs_month − t` is −1 month for 437 of 439 panel rows and −2 for the two shutdown dates listed above (1996-01-31, 2025-11-30). `indpro_obs_month − t` and `unrate_obs_month − t` are printed the same way at the end of the block.

### Step 1.7 — `pytest -q` at the end of the step

```
$ pytest -q
............................                                             [100%]
28 passed in 2.59s
```

## Tests run

Final state of the repo (after the step 1.7 commit), from the repo root. `.venv` was created with `uv venv --python 3.11` and `uv pip install -e .` from `pyproject.toml`; no library outside `pyproject.toml` was installed and `uv.lock` was not created.

```
$ pytest -q
............................                                             [100%]
28 passed in 2.74s
```

The 28 tests, by file: `test_config.py` (2: `test_config_keys_round_trip`, `test_config_is_frozen_and_typed`), `test_run.py` (2: `test_sections_registered_in_order`, `test_unbuilt_section_raises`), `test_fred_client.py` (5: `test_missing_api_key_raises`, `test_pull_id_format`, `test_write_raw_never_overwrites`, `test_manifest_appends`, `test_sha256_of_overrides_parquet_digest`), `test_market.py` (7: `test_lookback_rule` × 5 cases, `test_no_future_observation`, `test_index_runs_from_start_to_last_month_end`), `test_alfred.py` (4: `test_pinned_vintages_start_before_1995`, `test_value_used_has_realtime_start_le_t`, `test_no_release_before_t_gives_all_nan`, `test_cpi_obs_month_is_t_minus_1`), `test_french.py` (3: `test_parse_stops_at_first_blank_line`, `test_2010_01_row_matches_site`, `test_sample_end_is_momentum_last_month`), `test_project1.py` (2: `test_absent_returns_empty_schema_and_logs`, `test_present_round_trip`), `test_asof_panel.py` (3: `test_schema_matches_section_10`, `test_planted_future_market_value_is_ignored`, `test_lags_come_from_same_vintage`). Every test `PLAN.md` names is present; `test_unbuilt_section_raises`, `test_sha256_of_overrides_parquet_digest`, the fifth `test_lookback_rule` case (`latest_wins`), `test_index_runs_from_start_to_last_month_end` and `test_no_release_before_t_gives_all_nan` are additional. No test touches the network; tests that need pulled data read the committed pinned files and would fail, not skip, if absent.

Two test-side mistakes were made and fixed before their step's commit, recorded for completeness: in `test_value_used_has_realtime_start_le_t` an `iterrows` loop upcast a float NaN to `NaT` (pandas row upcasting on a row of two datetimes and a NaN), replaced by `itertuples`; in `test_planted_future_market_value_is_ignored` an extra assertion of mine expected the planted 1e6 to surface at a later month-end, which cannot happen in a daily series with a value every day, and was replaced by asserting the clean panel never contains it. Neither touched the code under test.

## Runtime per step

| step | wall-clock | machine | notes |
|---|---|---|---|
| env | ~3:00 | Windows 11 Home, 16 GB, Python 3.11.15 via uv | `.venv` creation and `uv pip install -e .`, before step 1.1 (17 Sep) |
| 1.1 | 1:07 | same | 17 Sep 14:13:22Z to 14:14:29Z |
| 1.2 | 0:32 | same | 17 Sep 14:15:37Z to 14:16:09Z |
| 1.5 | 1:39 | same | 18 Sep 08:29:56Z to 08:31:35Z, including the two downloads |
| 1.6 | 0:15 | same | 18 Sep 08:31:56Z to 08:32:11Z |
| 1.3 | 1:37 | same | 22 Sep 00:47:52Z to 00:49:29Z, including the seven FRED pulls |
| 1.4 | ~9:00 active | same | started 22 Sep 00:50:18Z; the three ALFRED pulls finished by 00:50:53Z; the session then sat idle between turns until ~09:52Z (machine clock; commit 10:00:51Z). Active work — tests, as-of build, evidence, commit — was about 9 minutes. The clock span (9 h 10 m) is an idle gap, not runtime. |
| 1.7 | 1:32 | same | 22 Sep 10:01:25Z to 10:02:57Z |

No step's active time approached the 20-minute threshold (`run.step_timeout_minutes`). `python -m regime.run --section 1` without `--pull` takes about 20 seconds.

## Not verified

- `parse_french_csv` on a CSV whose monthly block is not the first block: both live files have the monthly block first and the parser takes the first comma-led header line; a differently ordered file would be parsed wrongly and is not guarded against.
- `diff_french` against a genuinely different snapshot: only the empty self-diff was exercised; no second pull exists yet.
- `write_raw` with a frame that has no `date` column writes blank `first_date`/`last_date`; every frame the plan specifies has a `date` column, so the branch is untested.
- `asof_from_releases` when a release row carries a missing value (FRED `.`) for a month that had an earlier non-missing release: the pivot-and-forward-fill implementation would carry the earlier value forward past the missing release. The one such row in the pinned data (CPIAUCSL, October 2025, released 2025-12-18 as missing) has no earlier release, so it is NaN as required; no case of a missing release after a valid one was found or tested.
- `month_end_market` for a series whose raw history ends before `sample.end` (DTWEXM): the series simply stops at its last month-end; the panel's reindex supplies the NaN. Tested only via the live data (`dtwexm` 79 NaN from 2020-01), not by a synthetic test.
- The market lookback rule against the live series was checked only through the NaN pattern (no NaN in DGS10, DGS2, DCOILWTICO, VIXCLS; the expected gaps in DTWEXBGS and T10YIE); no month-end value was checked by hand against the FRED website.
- `test_cpi_obs_month_is_t_minus_1` reads the as-of parquet in `data/interim/` if present and rebuilds it otherwise; a stale interim file from a different pull is not detected (the file name carries the pinned `pull_id`, so only a file with the same id could be stale).
- `run.py --pull` pulls market and vintage series through two `FredClient`s (two `pull_id`s, matching the two config keys) and French through `pull_french`; the full `--pull` path was exercised only in pieces (market at step 1.3, vintages at 1.4 by direct calls, French at 1.5), never as one `--pull` run end to end after 1.7.
- Line endings: files were written with LF and git converts to CRLF on checkout (`core.autocrlf` is on in this checkout, as it was for session 0). Not verified that this matters to anything.

## Open questions

None appended to `decisions/OPEN.md`. The 100,000-row cap of `get_series_all_releases` was checked and not hit (max 39,362 rows, INDPRO), so the plan's truncation contingency did not arise.

## Files changed

Step 1.1 (`ff81cbf`):
- added `regime/__init__.py`, `regime/config.py`, `regime/run.py`
- added docstring-only modules `regime/data/__init__.py`, `regime/data/fred.py`, `regime/data/alfred.py`, `regime/data/french.py`, `regime/data/project1.py`, `regime/data/asof.py`, `regime/features.py`, `regime/models/__init__.py`, `regime/models/rules.py`, `regime/models/hmm_numpy.py`, `regime/models/hmm.py`, `regime/models/gmm.py`, `regime/models/anchor.py`, `regime/conditional.py`, `regime/strategy.py`, `regime/charts.py`, `regime/tables.py`
- added `tests/test_config.py`, `tests/test_run.py`; deleted `tests/test_placeholder.py`

Step 1.2 (`1fb67ac`):
- modified `regime/data/fred.py` (`FredClient`, `new_pull_id`, `raw_path`, `append_manifest_row`, `write_raw`); added `tests/test_fred_client.py`

Step 1.5 (`ff9e132`):
- modified `regime/data/french.py` (`parse_french_csv`, `join_french`, `pull_french`, `load_french`, `momentum_frame`, `sample_end`, `diff_french`)
- added `tests/test_french.py`; modified `tests/test_config.py` (tightened `french_pull_id`, `sample_end`)
- modified `config.toml` (`french.pull_id`, `sample.end`)
- added `data/raw/french/ff5_20260918T083009Z.zip`, `data/raw/french/mom_20260918T083009Z.zip`, `data/raw/french/factors_20260918T083009Z.parquet`, `data/raw/manifest.csv` (3 rows)

Step 1.6 (`61ac025`):
- modified `regime/data/project1.py` (`load_project1`); added `tests/test_project1.py`

Step 1.3 (`89b9992`):
- modified `regime/data/fred.py` (`FredClient.pull_market`, `month_end_from_daily`, `month_end_market`), `regime/run.py` (`section_1`: market pulls with `--pull`, month-end series)
- added `tests/test_market.py`; modified `tests/test_config.py` (tightened `fred_market_pull_id`)
- modified `config.toml` (`fred.market_pull_id`)
- added `data/raw/fred/{DGS10,DGS2,DTWEXBGS,DTWEXM,DCOILWTICO,VIXCLS,T10YIE}_20260922T004835Z.parquet`; `data/raw/manifest.csv` +7 rows

Step 1.4 (`c468292`):
- modified `regime/data/fred.py` (`FredClient.pull_vintages`), `regime/data/alfred.py` (`asof_from_releases`, `load_releases`, `asof_path`, `build_asof`), `regime/run.py` (`section_1`: vintage pulls with the pre-1995 check, as-of tables)
- added `tests/test_alfred.py`; modified `tests/test_config.py` (tightened `fred_vintage_pull_id`)
- modified `config.toml` (`fred.vintage_pull_id`)
- added `data/raw/alfred/{CPIAUCSL,INDPRO,UNRATE}_20260922T005041Z.parquet`; `data/raw/manifest.csv` +3 rows

Step 1.7 (`327b8db`):
- modified `regime/data/asof.py` (`assemble_panel`, `build_asof_panel`), `regime/run.py` (`section_1`: French `--pull` with `diff_french`, `load_project1`, panel write)
- added `tests/test_asof_panel.py`

Review: `review/section_1.md` (this file; earlier versions at `7cbe025` and `24322a6`).

Generated and not committed (ignored): `data/interim/asof_{CPIAUCSL,INDPRO,UNRATE}_20260922T005041Z.parquet`, `data/processed/asof_panel.parquet`, `.venv/`. Untracked and untouched: `Project Outline/` (present before session 1).

## Reviewer reads

1. `regime/data/alfred.py` — `asof_from_releases`: the pivot on `realtime_start` × `obs_month`, the forward fill down release dates, and `searchsorted(t, side="right") - 1` selecting the last release date ≤ t; the "no fallback to the current vintage" branch.
2. `regime/data/asof.py` — `_revised_columns`: m is the latest non-NaN month in the as-of-t vintage and the lag is read from the same vintage by `MonthEnd(lag)`; the column order against section 10.
3. `regime/data/fred.py` — `month_end_from_daily`: the window `t − lookback_days ≤ date ≤ t`, and that `pull_market`/`pull_vintages` share the client's single `pull_id`.
4. `tests/test_asof_panel.py` and `tests/test_alfred.py` — that the planted-future and same-vintage tests exercise the rule (the value encodes its own `realtime_start`).
5. `config.toml` lines 20, 32, 33, 39 against `data/raw/manifest.csv` — the four pinned keys match the pulls.
6. This file, "Step 1.7" — the single `cpi_m3` NaN (October 2025 CPI never published) and the two shutdown dates in the t − 1 statistic, both consequences of the as-of rule rather than faults.
