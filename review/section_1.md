# Review — Section 1: Foundation and data

## Section

This is Section 1 of `PLAN.md` (steps 1.1 to 1.7): the package skeleton and `Config`, the `FredClient` raw-write conventions, the FRED market pulls and `month_end_market`, the ALFRED vintage pulls and `build_asof`, the French pull, the project 1 adapter, and the as-of panel `data/processed/asof_panel.parquet`. **The section is incomplete: steps 1.3, 1.4 and 1.7 are not built because `FRED_API_KEY` is not set.** It is absent from the process environment, the User scope and the Machine scope (`[Environment]::GetEnvironmentVariable("FRED_API_KEY", "User")` and `("FRED_API_KEY", "Machine")` are both `$null`), and there is no `.env` in the repo. The session first stopped before 1.3 as instructed (17 September); on resumption (18 September) the two steps that do not depend on FRED — 1.5 (French, plain HTTP download) and 1.6 (no network) — were built and committed out of numeric order, following rule 1's principle of finishing every step that does not depend on the blocker. Step 1.7 depends on the market and vintage files from 1.3 and 1.4 and was not started.

**API needed to finish the section: a FRED API key**, exported as the environment variable `FRED_API_KEY` (free from https://fred.stlouisfed.org/docs/api/api_key.html). With it set, a new session runs steps 1.3, 1.4 and 1.7 in that order and pushes.

## Steps completed

- `step 1.1: package skeleton, Config with load_config, section registry and the key round-trip test` — `ff81cbf`
- `step 1.2: FredClient with one pull_id per session, write_raw that never overwrites, and the append-only manifest` — `1fb67ac`
- `step 1.5: French pull, parse, checksum; pin french.pull_id 20260918T083009Z and sample.end 2026-07-31` — `ff9e132`
- `step 1.6: project 1 adapter that logs project1: absent and returns the empty (date, factor, ret) schema` — `61ac025`

Not started: 1.3, 1.4, 1.7 (need `FRED_API_KEY`).

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

Every `config.toml` array is a `tuple`; `hmm_tol`, `gmm_tol`, `hmm_assigned_threshold`, `hmm_anchor_tie_tolerance`, `bootstrap_p_low`, `bootstrap_p_high` and `strategy_headline_eta` are `float`; at step 1.1 the three `*_pull_id` fields and `sample_end` were the empty `str` placeholders; `french_pull_id` and `sample_end` were filled at step 1.5, and the two FRED pull ids stay empty until steps 1.3 and 1.4.

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

`SECTIONS` has keys 1 … 7 in order (`tests/test_run.py::test_sections_registered_in_order`). All seven sections raise `NotImplementedError` until built; section 1 gets its body at step 1.3.

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

The momentum file (1195 rows from 1927-01) starts before the 5-factor file (757 rows from 1963-07); the inner join has 757 rows, 1963-07-31 to 2026-07-31. `diff_french` is empty on this, the first pull: there is no earlier snapshot, so the self-diff is shown to demonstrate the empty schema.

### Step 1.5 — `pytest -q` at the end of the step

```
$ pytest -q
............                                                             [100%]
12 passed in 1.41s
```

### Step 1.6 — the `project1: absent` log line and the returned frame's `dtypes`

`PLAN.md` asks for the log line from `python -m regime.run --section 1`; section 1's body is owned by step 1.3 and still raises `NotImplementedError("section 1 not built")`, so the line was captured from a direct `load_project1()` call with the same logger configuration `run.py` uses:

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

## Tests run

Final state of the repo (after the step 1.6 commit), from the repo root. `.venv` was created with `uv venv --python 3.11` and `uv pip install -e .` from `pyproject.toml`; no library outside `pyproject.toml` was installed and `uv.lock` was not created.

```
$ pytest -q
..............                                                           [100%]
14 passed in 0.78s
```

The fourteen tests: `tests/test_config.py::test_config_keys_round_trip`, `tests/test_config.py::test_config_is_frozen_and_typed`, `tests/test_run.py::test_sections_registered_in_order`, `tests/test_run.py::test_unbuilt_section_raises`, `tests/test_fred_client.py::test_missing_api_key_raises`, `tests/test_fred_client.py::test_pull_id_format`, `tests/test_fred_client.py::test_write_raw_never_overwrites`, `tests/test_fred_client.py::test_manifest_appends`, `tests/test_fred_client.py::test_sha256_of_overrides_parquet_digest`, `tests/test_french.py::test_parse_stops_at_first_blank_line`, `tests/test_french.py::test_2010_01_row_matches_site`, `tests/test_french.py::test_sample_end_is_momentum_last_month`, `tests/test_project1.py::test_absent_returns_empty_schema_and_logs`, `tests/test_project1.py::test_present_round_trip`. `test_unbuilt_section_raises` and `test_sha256_of_overrides_parquet_digest` are additional to the ones `PLAN.md` names; every named one is present. No test touches the network; the French tests read the committed pinned files.

## Runtime per step

| step | wall-clock | machine | notes |
|---|---|---|---|
| env | ~3:00 | Windows 11 Home, 16 GB, Python 3.11.15 via uv | `.venv` creation and `uv pip install -e .`, before step 1.1 |
| 1.1 | 1:07 | same | 14:13:22Z to 14:14:29Z |
| 1.2 | 1:40 | same | 14:14:29Z to 14:16:09Z |
| 1.5 | 1:26 | same, 18 Sep | 08:30:09Z to 08:31:35Z, including the two downloads |
| 1.6 | 0:36 | same, 18 Sep | 08:31:35Z to 08:32:11Z |
| 1.3, 1.4, 1.7 | — | | not started: `FRED_API_KEY` not set |

No step approached the 20-minute threshold (`run.step_timeout_minutes`).

## Not verified

- `FredClient` against the live FRED API: constructing `fredapi.Fred` makes no request, so `test_pull_id_format` passes with a dummy key; nothing proves the key or the endpoint.
- `pull_market`, `pull_vintages`, `month_end_market`, `build_asof`, `build_asof_panel`: not built.
- `fred.market_pull_id`, `fred.vintage_pull_id`: still `""`; the two-phase assertion in `tests/test_config.py` is tightened for `french_pull_id` and `sample_end` (step 1.5) and still loose for the two FRED keys.
- `diff_french` against a genuinely different snapshot: only the empty self-diff was exercised; no second pull exists yet.
- `parse_french_csv` on a CSV whose monthly block is not the first block: both live files have the monthly block first, and the parser takes the first comma-led header line, so a differently ordered file would be parsed wrongly and is not guarded against.
- The `project1: absent` line from `python -m regime.run --section 1` itself: section 1 is not built until step 1.3.
- The pre-1995 `realtime_start` check for the vintage series (step 1.4): not run.
- `python -m regime.run --section 1`: raises `NotImplementedError("section 1 not built")` by design until step 1.3; not exercised beyond that.
- `write_raw` with a frame that has no `date` column writes blank `first_date`/`last_date`; every frame the plan specifies has a `date` column, so the branch is untested.
- Line endings: the new files were written with LF and git reported it will convert them to CRLF on checkout (`core.autocrlf` is on in this checkout, as it was for session 0). Not verified that this matters to anything.

## Open questions

None appended to `decisions/OPEN.md`. The missing key is an environment fact, not a design choice, so it is reported here rather than as an `OPEN.md` item.

## Files changed

Step 1.1 (`ff81cbf`):
- added `regime/__init__.py`, `regime/config.py`, `regime/run.py`
- added docstring-only modules `regime/data/__init__.py`, `regime/data/fred.py`, `regime/data/alfred.py`, `regime/data/french.py`, `regime/data/project1.py`, `regime/data/asof.py`, `regime/features.py`, `regime/models/__init__.py`, `regime/models/rules.py`, `regime/models/hmm_numpy.py`, `regime/models/hmm.py`, `regime/models/gmm.py`, `regime/models/anchor.py`, `regime/conditional.py`, `regime/strategy.py`, `regime/charts.py`, `regime/tables.py`
- added `tests/test_config.py`, `tests/test_run.py`
- deleted `tests/test_placeholder.py`

Step 1.2 (`1fb67ac`):
- modified `regime/data/fred.py` (docstring only → `FredClient`, `new_pull_id`, `raw_path`, `append_manifest_row`, `write_raw`)
- added `tests/test_fred_client.py`

Step 1.5 (`ff9e132`):
- modified `regime/data/french.py` (docstring only → `parse_french_csv`, `join_french`, `pull_french`, `load_french`, `momentum_frame`, `sample_end`, `diff_french`)
- added `tests/test_french.py`; modified `tests/test_config.py` (tightened `french_pull_id` and `sample_end`)
- modified `config.toml` (`french.pull_id = "20260918T083009Z"`, `sample.end = "2026-07-31"`)
- added `data/raw/french/ff5_20260918T083009Z.zip`, `data/raw/french/mom_20260918T083009Z.zip`, `data/raw/french/factors_20260918T083009Z.parquet`, `data/raw/manifest.csv` (three rows)

Step 1.6 (`61ac025`):
- modified `regime/data/project1.py` (docstring only → `load_project1`)
- added `tests/test_project1.py`

This review: `review/section_1.md` (first written after step 1.2, rewritten after step 1.6).

Not committed and ignored: `.venv/` (`.gitignore`). Untracked and untouched: `Project Outline/` (present before this session).

## Reviewer reads

1. `regime/config.py` — the 86 field names against `config.toml`, the `float` coercion of int-valued TOML numbers, and that `KeyError` names the key in both directions.
2. `tests/test_config.py` — the two-phase comment on the `*_pull_id` / `sample_end` assertion; it is at the loose phase and steps 1.3 to 1.5 must each tighten one key in the same commit.
3. `regime/data/fred.py` — `pull_id` stamped once in `__init__`; `write_raw` raises `FileExistsError` before touching anything; the manifest is opened in append mode only; `sha256_of` overrides the parquet digest.
4. `regime/data/french.py` — `parse_french_csv` stops at the first blank line; `UMD` comes from `Mom`; the two zip manifest rows carry the zip digests; `sample_end` reads the momentum file.
5. `config.toml` lines 20 and 39 against `data/raw/manifest.csv` — the pinned `french.pull_id` and `sample.end` match the pull.
6. `regime/run.py` — the `SECTIONS` registry and that `--pull` reaches only section 1.
7. The first paragraph of this file — what is blocked and the API key needed to finish.
