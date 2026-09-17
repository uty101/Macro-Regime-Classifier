# PLAN.md — Macro Regime Classifier and Factor Timing

This is the sectioned build plan. One section per session, every step in the section in that session, one commit per step (`step X.Y: <one line>`), one review file per section (`review/section_X.md` from `review/TEMPLATE.md`). The rules that hold for every session are section 0 of `CLAUDE.md`; the timing convention is section 3 of `CLAUDE.md`; the twelve resolved conventions are `docs/CONVENTIONS_RESOLVED.md`. Every parameter is in `config.toml`. Where this plan fixes something the kickoff left implicit, it says so with "Fixed here:"; a session does not revisit those. Anything this plan does not fix is an `decisions/OPEN.md` item, not a choice.

Each step below states: what is built, the file it lives in, the signatures it implements, the test that proves it (file and test name), and the review evidence to attach.

## Conventions that apply to every step

- **Config access.** `Config` is a frozen dataclass with one field per `config.toml` key, named `<table>_<key>` (`hmm.n_iter` → `cfg.hmm_n_iter`). TOML arrays become tuples. Dates are ISO strings; code converts with `pd.Timestamp(...)`. No module reads `config.toml` except `regime/config.py`; every function that needs a parameter takes `cfg: Config`.
- **Seeds.** Restart i of any fit uses `random_state = cfg.run_seed + i`, i = 0 … n_restarts − 1. Every bootstrap uses `seed = cfg.run_seed`. Synthetic data in tests is generated with `np.random.default_rng(cfg.run_seed)`. Nothing uses default randomness.
- **Indices.** Every monthly frame is indexed by a `DatetimeIndex` of month-end decision dates named `date`, ascending, without duplicates, unless the schema says otherwise (the ALFRED long table). Returns are decimals.
- **Raw data.** Never overwritten. Every pull writes `data/raw/<source>/<series>_<pull_id>.parquet` with a `pulled_at` column and appends one row to `data/raw/manifest.csv`. Pinned snapshots are the `pull_id` keys in `config.toml`; the run always reads the pinned snapshot. `data/raw/` is committed in full (convention 14); `data/interim/` and `data/processed/` are regenerated and never committed.
- **Tests.** `pytest -q` from the repo root passes at the end of every step. Tests that need synthetic data build it in the test. Tests that need pulled data read the pinned `pull_id` from `config.toml` and fail (not skip) if the file is absent; the pinned raw files are committed, so a fresh clone passes `pytest` without a FRED key. No test touches the network.
- **Review evidence.** Raw rows via `to_string()` for every number. Runtime per step recorded; the threshold is `cfg.run_step_timeout_minutes` (20) per step, or `cfg.run_long_step_timeout_minutes` (60) for the steps in `cfg.run_long_steps` (3.4, 3.5, 3.6, 6.1, 6.2, 6.3). Over the threshold: stop and report (rule 10).
- **Out-of-sample window (fixed here).** Every conditional statistic, every bootstrap, the filtered-versus-smoothed gap and every backtest use decision dates from `sample.first_window_end` (2004-12-31) inclusive to the last decision date whose required future return exists, for every label source. Labels before `first_window_end` (rules and smoothed have them) are written to `regime_labels.csv` but consumed by nothing downstream, so the four sources are compared on identical dates.
- **Assigned (fixed here).** For every probabilistic source (HMM filtered, HMM smoothed, GMM) `assigned = max probability > hmm.assigned_threshold`; the hard label is the argmax. Rules labels are always assigned. A decision date with no probability row (dropped for NaN features, or before the first refit) is unassigned with label NaN.

---

## Section 1 — Foundation and data

### Step 1.1 — package skeleton, `load_config`, and the config-key round-trip test

**Builds.** The `regime` package with every module named in the layout, each containing only a module docstring until the step that owns it: `regime/__init__.py`, `regime/config.py`, `regime/run.py`, `regime/data/__init__.py`, `regime/data/fred.py`, `regime/data/alfred.py`, `regime/data/french.py`, `regime/data/project1.py`, `regime/data/asof.py`, `regime/features.py`, `regime/models/__init__.py`, `regime/models/rules.py`, `regime/models/hmm_numpy.py`, `regime/models/hmm.py`, `regime/models/gmm.py`, `regime/models/anchor.py`, `regime/conditional.py`, `regime/strategy.py`, `regime/charts.py`, `regime/tables.py`. `tests/test_placeholder.py` is deleted in this step and replaced by `tests/test_config.py`.

`regime/config.py`: `Config`, a `@dataclass(frozen=True)` with one field per `config.toml` key named `<table>_<key>`, typed `int`, `float`, `str`, `bool` or `tuple`; `load_config(path: str = "config.toml") -> Config` reads the file with `tomllib`, converts lists to tuples, and raises `KeyError` naming the offending key if the TOML has a key with no field or a field has no key.

`regime/run.py`: `python -m regime.run [--section N] [--pull]`. Without `--section` it runs sections 1 to 7 in order; with it, one section. Each section is a function `section_N(cfg: Config, pull: bool = False) -> None` registered in `SECTIONS: dict[int, Callable]`; sections not yet built raise `NotImplementedError("section N not built")`. `--pull` is honoured only by section 1 and performs fresh raw pulls (new `pull_id`, new raw files, manifest rows) without changing `config.toml`; without it, section 1 rebuilds `data/processed/asof_panel.parquet` from the pinned `pull_id`s. Every section rebuilds its outputs from `data/processed/` deterministically, so running twice produces identical files (idempotent). Wall-clock seconds per section are appended to `outputs/tables/runtime.csv` (columns: run_started, section, seconds) in step 7.3; until then the registry only.

**Signatures.** `load_config(path: str = "config.toml") -> Config`.

**Tests.** `tests/test_config.py::test_config_keys_round_trip` — flattens `config.toml` with `tomllib` into the set `{f"{table}_{key}"}` and asserts it equals `{f.name for f in dataclasses.fields(Config)}`, in both directions, with the symmetric difference in the assertion message. `tests/test_config.py::test_config_is_frozen_and_typed` — `load_config()` returns a `Config`; assigning to a field raises `dataclasses.FrozenInstanceError`; `cfg.run_seed == 20260917`; list-valued keys are tuples. Two-phase behaviour, stated in a comment in the test: every field ending in `_pull_id` and the field `sample_end` are asserted to be `str` from step 1.1 on, and additionally non-empty once steps 1.3 to 1.5 have committed their values (the step that fills a key tightens the assertion for that key in the same commit). `tests/test_run.py::test_sections_registered_in_order` — `SECTIONS` has keys exactly 1 … 7 in order.

**Review evidence.** `pytest -q` output; the field list of `Config` printed; the `python -m regime.run --section 2` output showing `NotImplementedError("section 2 not built")`.

### Step 1.2 — `FredClient` with raw-write conventions and manifest

**Builds.** `regime/data/fred.py`: `class FredClient` whose `__init__(self, cfg: Config)` reads `os.environ[cfg.fred_api_key_env]` and raises `RuntimeError("FRED_API_KEY not set")` if absent or empty, constructs `fredapi.Fred(api_key=...)`, and stamps `self.pull_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")` once at construction: one client is one pull session, every series pulled through it shares the `pull_id`, and every `pull_*` method returns it. The raw-write helper `write_raw(frame: pd.DataFrame, source: str, series: str, pull_id: str, cfg: Config, sha256_of: bytes | None = None) -> Path` writes `data/raw/<source>/<series>_<pull_id>.parquet` (creating the source folder), raises `FileExistsError` if the path already exists, and appends one row to `cfg.fred_manifest` with columns `source, series, pull_id, rows, first_date, last_date, sha256, pulled_at` — `sha256` is the SHA-256 of the parquet file bytes unless `sha256_of` (the downloaded zip bytes, for French) is given. The manifest is created with a header on first write and only ever appended. The `pulled_at` column of every raw frame is the UTC timestamp of the pull as an ISO string. `source ∈ {"fred", "alfred", "french"}`.

**Signatures.** `FredClient.__init__(self, cfg: Config)`; `write_raw(...) -> Path` as above.

**Tests.** `tests/test_fred_client.py::test_missing_api_key_raises` — `monkeypatch.delenv("FRED_API_KEY", raising=False)`; `FredClient(cfg)` raises `RuntimeError` with message exactly `"FRED_API_KEY not set"`. `tests/test_fred_client.py::test_pull_id_format` — with a dummy key set, `client.pull_id` matches `^\d{8}T\d{6}Z$`. `tests/test_fred_client.py::test_write_raw_never_overwrites` — in `tmp_path` (config replaced via `dataclasses.replace` to point `fred_raw_dir` and `fred_manifest` there), writing the same (source, series, pull_id) twice raises `FileExistsError` on the second write and the first file's bytes are unchanged. `tests/test_fred_client.py::test_manifest_appends` — two different writes produce a manifest with exactly two rows, the first row byte-identical to what it was after the first write, and `sha256` equal to `hashlib.sha256(path.read_bytes()).hexdigest()`.

**Review evidence.** `pytest -q` output; the manifest header line; the exception text captured.

### Step 1.3 — market series pulls and `month_end_market` with the 10-day lookback test

**Builds.** `FredClient.pull_market(self, series_id: str) -> str`: `fred.get_series(series_id)` for the full history, frame `(date, value, pulled_at)` with FRED's missing markers as NaN, written through `write_raw(source="fred")`, returns `self.pull_id`. `month_end_market(series_id: str, pull_id: str, cfg: Config) -> pd.Series`: reads `data/raw/fred/<series_id>_<pull_id>.parquet`; for every month-end t from `cfg.sample_start` to the last month-end ≤ the last raw observation date, the value is the last non-missing observation with `t − cfg.fred_lookback_days days ≤ date ≤ t`, else NaN; returns a `Series` named `series_id` indexed by `date`. The pure helper `month_end_from_daily(daily: pd.Series, start: pd.Timestamp, lookback_days: int) -> pd.Series` does the work so the test needs no file. Section 1 of `run.py` with `--pull` pulls all `cfg.fred_market_series` through one client; the session then writes that `pull_id` into `fred.market_pull_id` in `config.toml` in this step's commit, and the commit includes the raw parquet files it pulled and the manifest rows.

**Signatures.** `pull_market(self, series_id: str) -> str`; `month_end_market(series_id: str, pull_id: str, cfg: Config) -> pd.Series`.

**Tests.** `tests/test_market.py::test_lookback_rule` — synthetic daily series; parametrised cases: an observation dated t is used; the last observation dated t − 10 days is used; the last observation dated t − 11 days gives NaN; a NaN on t with a value on t − 1 uses t − 1. `tests/test_market.py::test_no_future_observation` — a huge value dated t + 1 day leaves the value at t unchanged and is not used for t.

**Review evidence.** Manifest rows for the seven series (`to_string()`); `head(3)` and `tail(3)` of each month-end series; a table (series, n_months, n_nan, nan_dates) with the NaN dates listed, expected to include DTWEXBGS before 2006-01, DTWEXM after 2020-01, T10YIE before 2003-01, VIXCLS before 1990-01.

### Step 1.4 — ALFRED vintage pulls and `build_asof`

**Builds.** `FredClient.pull_vintages(self, series_id: str) -> str`: `fred.get_series_all_releases(series_id)`, frame `(date, realtime_start, value, pulled_at)` where `date` is the observation month (month-start as FRED reports it, stored as given) and `realtime_start` the release date, all rows stored raw, written through `write_raw(source="alfred")`, returns `self.pull_id`. `regime/data/alfred.py`: `build_asof(series_id: str, pull_id: str, cfg: Config) -> pd.DataFrame`, long with columns `decision_date, obs_month, value` where `obs_month` is the month-end of the observation month; for each decision date t from `cfg.sample_start` to the last month-end ≤ the last `realtime_start` in the file, and each observation month m ≤ t, `value` is the value of the release with the greatest `realtime_start ≤ t`, and NaN if no release with `realtime_start ≤ t` exists for m. A decision date at which no release of any month has `realtime_start ≤ t` has all values NaN; there is no fallback to the current vintage. The pure helper `asof_from_releases(releases: pd.DataFrame, decision_dates: pd.DatetimeIndex) -> pd.DataFrame` does the work. The result is written to `data/interim/asof_<series_id>_<pull_id>.parquet`. Section 1 with `--pull` pulls all `cfg.fred_vintage_series` through one client; the session writes that `pull_id` into `fred.vintage_pull_id` in this step's commit, and the commit includes the raw vintage parquet files and the manifest rows.

After each `pull_vintages`, assert that the earliest `realtime_start` in the file is before 1995-01-01, and record per series the number of distinct `realtime_start` dates and the earliest one in the review evidence. If the earliest `realtime_start` is 1995-01-01 or later the pull is treated as truncated: append an `OPEN.md` item with the options (A: paginate `get_series_all_releases` by realtime range and concatenate; B: use the FRED observations endpoint directly with `realtime_start`/`realtime_end` windows through `requests`) and do not build the as-of table for that series until resolved.

**Signatures.** `pull_vintages(self, series_id: str) -> str`; `build_asof(series_id: str, pull_id: str, cfg: Config) -> pd.DataFrame`.

**Tests.** `tests/test_alfred.py::test_pinned_vintages_start_before_1995` — for every series in `cfg.fred_vintage_series`, the committed raw file `data/raw/alfred/<series>_<cfg.fred_vintage_pull_id>.parquet` has `realtime_start.min() < 1995-01-01`. `tests/test_alfred.py::test_value_used_has_realtime_start_le_t` — synthetic releases where every value encodes its own `realtime_start` (value = ordinal day of `realtime_start`), three releases per observation month; for every non-NaN row of `asof_from_releases`, `value ≤ decision_date.toordinal()` and `value == max realtime_start ≤ t` for that month, and rows for months whose first release is after t are NaN. `tests/test_alfred.py::test_cpi_obs_month_is_t_minus_1` — on the pinned CPIAUCSL as-of table, over all decision dates from `sample_start` to the last decision date in the table, the latest non-NaN `obs_month` at t equals the month-end of the month before t for at least 95% of decision dates; the failing dates are printed in the assertion message.

**Review evidence.** Manifest rows for the three series; per series, the number of distinct `realtime_start` dates and the earliest one; for each series, the rows of `build_asof` at decision dates 1995-01-31, 2008-12-31 and the last decision date for the last three observation months (`to_string()`); the count and list of decision dates with no available vintage per series; the percentage behind the 95% claim with the rows for every failing date; for INDPRO and UNRATE the same t − 1 statistic (no threshold, reported).

### Step 1.5 — French pull, parse, checksum, and the 2010-01 test

**Builds.** `regime/data/french.py`: `pull_french(cfg: Config) -> str` downloads `cfg.french_factors_url` and `cfg.french_momentum_url` with `requests`, saves the zip bytes as `data/raw/french/ff5_<pull_id>.zip` and `data/raw/french/mom_<pull_id>.zip` (never overwritten), parses each with `parse_french_csv(text: str) -> pd.DataFrame` — skip header lines until the first line that starts with a comma followed by the column names, then read rows keyed `YYYYMM` until the first blank line, divide by 100, index = month-end `Timestamp` named `date` — joins them (inner on date) into columns `Mkt-RF, SMB, HML, RMW, CMA, UMD, RF` with `UMD` from the momentum file's `Mom` column, writes `data/raw/french/factors_<pull_id>.parquet` through `write_raw(source="french", series="factors")`, and appends two further manifest rows (`series="ff5_zip"`, `series="mom_zip"`) whose `sha256` is that of the zip bytes. `pull_id` here comes from `datetime.now(timezone.utc)` at the start of the call, same format. `load_french(pull_id: str) -> pd.DataFrame` reads the parquet. `diff_french(new_pull_id: str, first_pull_id: str) -> pd.DataFrame` returns every (date, column, first_value, new_value) where the two snapshots differ, over their common dates; on any later pull section 1 prints it and the review file reports it. `sample_end(pull_id: str) -> pd.Timestamp` returns the last month in the momentum file. In this step's commit the session writes the pull's id into `french.pull_id` and the momentum file's last month into `sample.end` in `config.toml`; the commit includes the two zips, the factors parquet and the manifest rows.

**Signatures.** `pull_french(cfg: Config) -> str`; `load_french(pull_id: str) -> pd.DataFrame`; `parse_french_csv(text: str) -> pd.DataFrame`; `diff_french(new_pull_id: str, first_pull_id: str) -> pd.DataFrame`.

**Tests.** `tests/test_french.py::test_parse_stops_at_first_blank_line` — a synthetic CSV text with a header, three monthly rows, a blank line and an annual block; the parsed frame has exactly three rows, values divided by 100, month-end index. `tests/test_french.py::test_2010_01_row_matches_site` — `load_french(cfg.french_pull_id)` row for 2010-01-31 equals values the session hard-codes in the test after reading them by eye from the downloaded CSVs (all seven columns, `abs tol 1e-12` after the /100). `tests/test_french.py::test_sample_end_is_momentum_last_month` — `sample_end(cfg.french_pull_id)` equals `pd.Timestamp(cfg.sample_end)`.

**Review evidence.** Manifest rows for `factors`, `ff5_zip`, `mom_zip` with sha256; `head(3)`, the 2010-01-31 row, and `tail(3)` of the parsed frame; the last month of each file; `diff_french` output (empty on the first pull, stated).

### Step 1.6 — project 1 adapter

**Builds.** `regime/data/project1.py`: `load_project1(path: str | None = None) -> pd.DataFrame` reads `path` (default `load_config().outputs_project1_file`) if it exists and returns columns `date, factor, ret` (date as `Timestamp` month-end, factor `str`, ret `float`); otherwise logs `project1: absent` through `logging.getLogger("regime")` at INFO and returns an empty frame with those three columns and dtypes. The optional `path` argument exists only so the test can point at `tmp_path`; the no-argument call is the specified signature.

**Signatures.** `load_project1() -> pd.DataFrame`.

**Tests.** `tests/test_project1.py::test_absent_returns_empty_schema_and_logs` — `caplog` captures `project1: absent`; the result has the three columns and zero rows. `tests/test_project1.py::test_present_round_trip` — a two-row parquet written to `tmp_path` is read back equal.

**Review evidence.** The log line from `python -m regime.run --section 1` and the returned frame's `dtypes`.

### Step 1.7 — `build_asof_panel` with the planted-future-value and schema tests

**Builds.** `regime/data/asof.py`: `assemble_panel(market: dict[str, pd.Series], asof: dict[str, pd.DataFrame], cfg: Config) -> pd.DataFrame` (pure) and `build_asof_panel(cfg: Config) -> pd.DataFrame` which loads the pinned raw/interim files and calls it, writing `cfg.outputs_asof_panel`. Index: month-end decision dates from `cfg.sample_start` to `cfg.sample_end` inclusive, name `date`. Columns, in this order: `dgs10, dgs2, dtwexbgs, dtwexm, wti, vix, t10yie` (the `month_end_market` series for DGS10, DGS2, DTWEXBGS, DTWEXM, DCOILWTICO, VIXCLS, T10YIE), then `cpi_m, cpi_m3, indpro_m, indpro_m12, unrate_m, unrate_m12, cpi_obs_month, indpro_obs_month, unrate_obs_month`. For each revised series at decision date t: m = the latest observation month with a non-NaN value in the as-of-t vintage; `<s>_m` = that value; `<s>_m3` / `<s>_m12` = the value in the same as-of-t vintage for the observation month `cfg.features_cpi_lag` / `cfg.features_change_lag` observation months earlier (NaN if absent in that vintage); `<s>_obs_month` = m (NaT when the vintage is empty). `cpi_m3` uses `cpi_lag`; `indpro_m12` and `unrate_m12` use `change_lag`.

**Signatures.** `build_asof_panel(cfg: Config) -> pd.DataFrame`; `assemble_panel(...)` as above.

**Tests.** `tests/test_asof_panel.py::test_schema_matches_section_10` — the column list equals the sixteen names above in order, index is month-end, monotonic, named `date`, runs from `sample_start` to `sample_end`, the three `*_obs_month` columns are datetime, all others float. `tests/test_asof_panel.py::test_planted_future_market_value_is_ignored` — build the panel from synthetic inputs twice, the second time with a 1e6 value planted in the DGS10 daily series one day after decision date t; every panel value at rows ≤ t is identical to 1e-12. `tests/test_asof_panel.py::test_lags_come_from_same_vintage` — synthetic releases where the value encodes (obs_month, realtime_start); at t, `cpi_m` and `cpi_m3` both decode to a `realtime_start ≤ t` and the same vintage rule.

**Review evidence.** `panel.head(3)`, the rows for 2004-12-31 and `sample_end`, and a table (column, n_nan, first_valid, last_valid) via `to_string()`; the list of decision dates with NaN in any of `dgs10, dgs2, wti, vix, cpi_m, cpi_m3, indpro_m, indpro_m12` from `sample.features_from` onward; a table of `cpi_obs_month − t` in months, value counts.

---

## Section 2 — Features

### Step 2.1 — `build_raw_features` for all 10 columns

**Builds.** `regime/features.py`: `build_raw_features(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame`, same index as the panel, columns in this order: the eight `cfg.features_core` then the two `cfg.features_robustness`. Definitions (L = `cfg.features_change_lag` = 12 rows of the decision-date index; `shift(L)` on the panel):

| column | definition |
|---|---|
| `dgs10_level` | `dgs10` |
| `dgs10_chg12` | `dgs10 − dgs10.shift(L)` |
| `slope_2s10s` | `dgs10 − dgs2` |
| `cpi_3m_ann` | `((cpi_m / cpi_m3) ** (12 / cpi_lag) − 1) × 100` |
| `indpro_chg12` | `ln(indpro_m / indpro_m12) × 100` |
| `dollar_chg12` | `splice_dollar_chg12(panel, cfg)` (step 2.2) |
| `oil_chg12` | `ln(wti / wti.shift(L)) × 100` |
| `log_vix` | `ln(vix)` |
| `unrate_chg12` | `unrate_m − unrate_m12` |
| `breakeven_chg12` | `t10yie − t10yie.shift(L)` |

`splice_dollar_chg12` is written in this step (it is a column of the frame) and proven in step 2.2. Written to `cfg.outputs_features_raw`. Also `model_input(z: pd.DataFrame, cfg: Config, columns: tuple[str, ...] | None = None) -> pd.DataFrame`: restricts to `columns` (default `cfg.features_core`), restricts to dates ≥ `cfg.sample_features_from`, drops rows with any NaN, and writes the dropped dates and their NaN columns to `cfg.outputs_dropped_rows` (columns: date, missing).

**Signatures.** `build_raw_features(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame`.

**Tests.** `tests/test_features.py::test_raw_feature_definitions_by_hand` — a hand-built 14-row panel with small integers; each of the ten columns at the last row equals a number computed by hand and written literally in the test (`abs tol 1e-9`). `tests/test_features.py::test_lags_are_rows_not_days` — a panel whose `dgs10` equals its row number gives `dgs10_chg12 == 12` on every row from row 12 on, and NaN before.

**Review evidence.** `raw.loc["2004-12-31"]` and `raw.loc[cfg.sample_end]` via `to_string()`; the dropped-row table; `raw.tail(3)`.

### Step 2.2 — dollar splice with the continuity test

**Builds.** `splice_dollar_chg12(panel: pd.DataFrame, cfg: Config) -> pd.Series`: `m = ln(dtwexm / dtwexm.shift(L)) × 100`, `b = ln(dtwexbgs / dtwexbgs.shift(L)) × 100`; result equals `m` for dates < `cfg.features_dollar_splice_date` (2007-01-31) and `b` for dates ≥ it. Changes are spliced, never levels. DTWEXBGS begins in 2006-01, so its first 12-row change is valid at 2007-01-31: the splice date row is valid by construction.

**Tests.** `tests/test_features.py::test_dollar_splice_switches_exactly_at_splice_date` — synthetic panel; the row before the splice date equals the DTWEXM change, the splice date row equals the DTWEXBGS change. `tests/test_features.py::test_dollar_splice_has_no_level_jump` — synthetic panel where `dtwexm = 0.8 × dtwexbgs` on every date (same path, different base): the spliced series equals `ln(dtwexbgs / dtwexbgs.shift(L)) × 100` on every non-NaN row to 1e-12, which a level splice would fail.

**Review evidence.** The 12 rows either side of 2007-01-31 showing `dtwexm, dtwexbgs`, the two change series and `dollar_chg12` via `to_string()`; the value of `dollar_chg12` at 2007-01-31.

### Step 2.3 — `standardise` with the spike test and the first-window rule test

**Builds.** `standardise(raw: pd.DataFrame, cfg: Config) -> pd.DataFrame`: same columns as `raw`, rows from `cfg.sample_features_from`. For decision dates ≤ `cfg.sample_first_window_end`, `z = (x − μ_W) / σ_W` with μ, σ over the first window `[features_from, first_window_end]` (ddof = `cfg.features_ddof`), NaN rows excluded from the moments column by column. For dates after it, μ and σ at t are over `[features_from, t]` inclusive (expanding, ddof = 1). Robustness columns are standardised by the same rule over their non-NaN rows. Written once to `cfg.outputs_features_z`; nothing ever rewrites it during a run except this step's function.

**Signatures.** `standardise(raw: pd.DataFrame, cfg: Config) -> pd.DataFrame`.

**Tests.** `tests/test_features.py::test_spike_after_t_does_not_change_z_at_t` — synthetic raw frame from `default_rng(cfg.run_seed)`; parametrised over t ∈ {first_window_end, first_window_end + 1 row, a row 60 rows later, the last row − 1}; plant a spike of 10 × σ_W at row t + 1 in every column; every z at rows ≤ t is unchanged to 1e-12. (For t inside the first window the first-window moments change by construction — convention 2 — so those t are not in the parametrisation and the test says so in a comment.) `tests/test_features.py::test_first_window_rule` — for a row t inside the first window, z equals `(x_t − mean_W) / std_W(ddof=1)` computed independently in the test; for 2005-01-31, z equals the expanding statistic over `[features_from, 2005-01-31]`.

**Review evidence.** `z.loc[["1991-01-31", "2004-12-31", "2005-01-31"]]` and `z.tail(3)` via `to_string()`; per-column mean and std of z over the first window (expected 0 and 1 to 1e-12, printed with the rows that produced them counted); the dropped-row table from `model_input`.

### Step 2.4 — feature sanity table and review chart

**Builds.** `regime/tables.py`: `feature_sanity(raw: pd.DataFrame, cfg: Config) -> pd.DataFrame` with columns `feature, year, min, max, n_nan`, one row per feature × calendar year, written to `outputs/tables/feature_sanity.csv`. `regime/charts.py`: `features_review_chart(raw: pd.DataFrame, path: str, cfg: Config) -> None`, a 10-panel (5 × 2) time-series figure of the raw features from `sample_features_from` to `sample_end`, PNG at `cfg.outputs_chart_dpi`, written to `outputs/charts/features_review.png`. Section 2 of `run.py` runs 2.1–2.4 from `data/processed/asof_panel.parquet`.

**Tests.** `tests/test_tables.py::test_feature_sanity_shape` — on a synthetic raw frame spanning three years, the table has 10 × 3 rows, the five columns, and `n_nan` matching a hand count. `tests/test_charts.py::test_features_review_chart_written` — writes to `tmp_path`, file exists and is larger than 10 kB.

**Review evidence.** The full `feature_sanity.csv` via `to_string()`; the chart embedded or linked; a sentence per feature on whether its range is plausible (e.g. `cpi_3m_ann` in 2008-Q4 negative, `log_vix` peak in 2008-11 and 2020-03).

---

## Section 3 — Classifiers

### Step 3.1 — `rules_labels`

**Builds.** `regime/models/rules.py`: `rules_labels(raw: pd.DataFrame, cfg: Config) -> pd.Series` (int, name `rules_label`, index = rows of `raw` from `sample_features_from` where both features are non-NaN). `growth_up = raw[cfg.rules_growth_feature] > expanding_median`, `inflation_up = raw[cfg.rules_inflation_feature] > expanding_median`, where the expanding median follows the first-window rule: for t ≤ `first_window_end` the median of the first window, after it the median over `[features_from, t]` inclusive. Labels: 0 = growth down & inflation down; 1 = growth up & inflation down; 2 = growth down & inflation up; 3 = growth up & inflation up. Written to `data/processed/rules_labels.parquet`.

**Tests.** `tests/test_rules.py::test_quadrant_labels_by_hand` — a synthetic raw frame where the medians are known; the labels at four chosen rows are 0, 1, 2, 3. `tests/test_rules.py::test_first_window_median_rule` — a row inside the first window is labelled against the whole-window median, a row after it against the expanding median (both computed independently in the test).

**Review evidence.** Label value counts over the out-of-sample window; the labels for 2008-06-30 to 2009-06-30 alongside the two features and their medians via `to_string()`.

### Step 3.2 — `forward_filter` and its two-part test against hmmlearn

**Builds.** `regime/models/hmm_numpy.py`: `forward_filter(y, startprob, transmat, means, covars) -> tuple[np.ndarray, float]` — scaled forward recursion: log emission densities from `scipy.stats.multivariate_normal.logpdf` per state (`covars` shape (K, d, d)); at each row, `α_t ∝ (α_{t−1} @ transmat) × exp(logB_t − max_t)`, normalised, with the log normalisers summed into the log-likelihood; returns the (T × K) filtered probabilities and the log-likelihood. Row 0 uses `startprob`.

**Tests.** `tests/test_hmm_numpy.py::test_loglik_matches_hmmlearn_score` — synthetic 3-state data, d = 3, T = 400 from `default_rng(cfg.run_seed)`; fit `GaussianHMM(n_components=3, covariance_type="full", random_state=cfg.run_seed, n_iter=cfg.hmm_n_iter, tol=cfg.hmm_tol)`; `abs(loglik − model.score(y)) < 1e-6`. `tests/test_hmm_numpy.py::test_final_row_matches_predict_proba` — the last row of the filtered probabilities equals the last row of `model.predict_proba(y)` to 1e-8; no earlier row is compared (convention 6, stated in a comment).

**Review evidence.** `pytest -q` output; the two numbers side by side.

### Step 3.3 — `select_k` and `bic_by_k.csv`

**Builds.** `regime/models/hmm.py`: `HMMParams` (frozen dataclass: `startprob, transmat, means, covars, K, refit_date, loglik, converged`) and `fit_hmm(z: np.ndarray, K: int, cfg: Config, refit_date: pd.Timestamp) -> tuple[HMMParams, pd.DataFrame]` — `cfg.hmm_n_restarts` fits of `GaussianHMM(n_components=K, covariance_type=cfg.hmm_covariance_type, n_iter=cfg.hmm_n_iter, tol=cfg.hmm_tol, init_params=cfg.hmm_init_params, params=cfg.hmm_params, random_state=cfg.run_seed + i)`; the restarts table has columns `restart, seed, loglik, n_iter, converged` where `loglik = model.score(z)` on the training rows, `n_iter = model.monitor_.iter`, `converged = model.monitor_.converged`; the kept restart is the highest `loglik`; `HMMParams.covars` is `model.covars_` (K, d, d). `fit_hmm` is built in this step because `select_k` needs it; step 3.4 adds anchoring and the expanding protocol. `select_k(z_first_window: pd.DataFrame, cfg: Config) -> tuple[int, pd.DataFrame]`: for K in `cfg.hmm_k_grid`, `fit_hmm` on the model-input rows of the first window, `BIC = −2 loglik + m log T` with `m = K(K−1) + Kd + Kd(d+1)/2`, T = rows; table columns `K, loglik, m, T, bic, converged`; `primary_K = argmin BIC over cfg.hmm_k_candidates`; writes `outputs/tables/bic_by_k.csv`, `data/processed/primary_k.txt` (the integer), and `outputs/tables/hmm_restarts_selectk_K<K>.csv` per K.

**Signatures.** As in the kickoff, section 10.

**Tests.** `tests/test_hmm.py::test_bic_parameter_count` — `n_params(K=3, d=8) == 138`, `n_params(K=4, d=8) == 188`, and `bic(loglik=−1000.0, m=138, T=168) == 2000 + 138 log 168` to 1e-9. `tests/test_hmm.py::test_select_k_argmin_over_candidates_only` — with a stubbed BIC table where K = 2 is the global minimum and K = 4 < K = 3, `primary_K == 4`. `tests/test_hmm.py::test_fit_hmm_restarts_are_seeded_and_deterministic` — synthetic data; the restarts table has `n_restarts` rows with `seed == cfg.run_seed + restart`; two calls give identical `loglik` columns.

**Review evidence.** `bic_by_k.csv` via `to_string()`; `primary_k.txt`; per K the restarts table sorted by loglik with the kept row marked and the number of non-converged restarts.

### Step 3.4 — `fit_hmm`, `anchor`, and `run_expanding_hmm` with restart tables, `state_counts.csv`, `anchor_agreement.csv`

**Builds.** `regime/models/anchor.py`: `anchor_permutation(means: np.ndarray, feature_names: list[str], cfg: Config) -> np.ndarray` — order states by ascending mean of `cfg.hmm_anchor_feature`; walk the sorted list and, wherever two adjacent means differ by less than `cfg.hmm_anchor_tie_tolerance`, order that pair by ascending mean of `cfg.hmm_anchor_tiebreak_feature`; return `perm` such that new state j = old state `perm[j]`. `anchor(params: HMMParams, feature_names: list[str], cfg: Config) -> tuple[HMMParams, np.ndarray]` applies it: `startprob[perm]`, `transmat[perm][:, perm]`, `means[perm]`, `covars[perm]`. `hungarian_agreement(prev_means: np.ndarray, new_means: np.ndarray) -> tuple[bool, np.ndarray]` — `linear_sum_assignment` on the Euclidean distance matrix between previous and new anchored means; `agrees = assignment is the identity`.

`run_expanding_hmm(z: pd.DataFrame, K: int, cfg: Config) -> tuple[pd.DataFrame, list[HMMParams]]`: refit dates R = `first_window_end` and every `cfg.hmm_refit_month` month-end after it with date ≤ `cfg.sample_end`. For each D in R: `fit_hmm` on model-input rows `[features_from, D]`, `anchor`, write `outputs/tables/hmm_restarts_<D:%Y-%m-%d>.csv`; count training rows per state (argmax of `model.predict_proba` on the training rows under the anchored parameters) into `outputs/tables/state_counts.csv` (columns `refit_date, state, n_rows, degenerate` with `degenerate = n_rows < cfg.hmm_min_state_rows`); for every D after the first, `hungarian_agreement` against the previous refit's anchored means into `outputs/tables/anchor_agreement.csv` (columns `refit_date, agrees, assignment`). Then for every model-input decision date t with D ≤ t < next D: `forward_filter` on the rows `[features_from, t]` under D's anchored parameters, last row → `p0 … p{K−1}`. Returned frame: index `date`, columns `p0 … p{K−1}, refit_date`, written to `cfg.outputs_filtered_probs` (with `date` as a column in the CSV). Rows exist only for model-input dates ≥ `first_window_end`. Non-converged kept restarts are reported, `n_iter` is not increased; degenerate states and disagreements are reported, not corrected. `hard_labels(probs: pd.DataFrame, cfg: Config) -> pd.DataFrame` (columns `date, label, assigned`) applies argmax and `assigned_threshold`.

**Tests.** `tests/test_anchor.py::test_anchor_orders_by_dgs10_chg12_and_permutes_consistently` — synthetic 3-state params with unsorted means; after `anchor` the anchor-feature means are ascending, and `transmat` equals `T[perm][:, perm]` with the same `perm` used on `means`, `covars`, `startprob`. `tests/test_anchor.py::test_anchor_tiebreak_within_tolerance` — two states with anchor-feature means 0.05 apart are ordered by `cpi_3m_ann`. `tests/test_anchor.py::test_hungarian_identity_and_swap` — identical means → `(True, [0,1,2])`; two swapped means → `(False, [1,0,2])`. `tests/test_hmm.py::test_refit_parameters_apply_from_refit_date_inclusive` — on synthetic z with two refit dates (config replaced to a short sample), `refit_date` at t = D equals D, and at t = D − 1 row equals the previous refit. `tests/test_hmm.py::test_filtered_is_fresh_full_history_pass` — for a t immediately after a refit boundary, the returned row equals `forward_filter(z[features_from : t], params_in_force)[0][-1]` recomputed in the test to 1e-12.

**Review evidence.** The list of refit dates; `state_counts.csv` and `anchor_agreement.csv` in full via `to_string()`; for each refit the kept restart's `loglik, n_iter, converged` and the count of non-converged restarts; `filtered_probs.head(3)` and `.tail(3)`; label value counts and the unassigned count; wall-clock of the whole step (rule 10).

### Step 3.5 — `run_smoothed_hmm`

**Builds.** `run_smoothed_hmm(z: pd.DataFrame, K: int, cfg: Config) -> tuple[pd.DataFrame, HMMParams]`: one `fit_hmm` on all model-input rows `[features_from, sample_end]` with `refit_date = sample_end`, `anchor`, restarts to `outputs/tables/hmm_restarts_smoothed.csv`, `model.predict_proba` on the full sequence under the anchored parameters → columns `p0 … p{K−1}`, index `date`, written to `cfg.outputs_smoothed_probs`. Used only for the hindsight comparison.

**Tests.** `tests/test_hmm.py::test_smoothed_rows_sum_to_one_and_cover_all_model_input_dates`. `tests/test_hmm.py::test_smoothed_final_row_equals_forward_filter_final_row` — under the same parameters, the last smoothed row equals the last row of `forward_filter` on the same sequence to 1e-8 (convention 6).

**Review evidence.** The kept restart's `loglik, n_iter, converged`; `smoothed_probs.head(3)`/`.tail(3)`; smoothed versus filtered hard-label agreement rate over the out-of-sample window with the confusion table via `to_string()`.

### Step 3.6 — `run_expanding_gmm`

**Builds.** `regime/models/gmm.py`: `run_expanding_gmm(z: pd.DataFrame, K: int, cfg: Config) -> pd.DataFrame` — same refit dates as 3.4; at each D, `cfg.gmm_n_restarts` fits of `GaussianMixture(n_components=K, covariance_type=cfg.gmm_covariance_type, n_init=cfg.gmm_n_init, max_iter=cfg.gmm_max_iter, tol=cfg.gmm_tol, random_state=cfg.run_seed + i)` on model-input rows `[features_from, D]`, kept by highest `lower_bound_`, restarts table (`restart, seed, lower_bound, n_iter, converged`) to `outputs/tables/gmm_restarts_<D>.csv`; anchoring by `anchor_permutation` applied to `means_`, `covariances_`, `weights_`; "filtered" probability at t (D ≤ t < next D) = `predict_proba(z_t)` under the anchored parameters in force. Frame: index `date`, columns `p0 … p{K−1}, refit_date`, written to `cfg.outputs_gmm_filtered_probs`.

**Tests.** `tests/test_gmm.py::test_gmm_restarts_seeded_and_deterministic`. `tests/test_gmm.py::test_gmm_probability_is_predict_proba_under_params_in_force` — recomputed in the test for a t after a refit boundary, equal to 1e-12.

**Review evidence.** GMM label value counts; agreement with HMM filtered labels (confusion table); the kept lower bounds per refit.

### Step 3.7 — transition matrices, `expected_duration.csv`, `param_drift.csv`

**Builds.** `regime/tables.py`: `write_hmm_tables(params: list[HMMParams], feature_names: list[str], cfg: Config) -> None` writes, per refit, `outputs/tables/transition_matrix_<D>.csv` (rows from-state, columns to-state, anchored); `outputs/tables/expected_duration.csv` (columns `refit_date, state, expected_duration` = 1 / (1 − A_kk)); `outputs/tables/param_drift.csv` (columns `refit_date, state, feature, mean, variance`, variance = diagonal of the anchored covariance). Section 3 of `run.py` runs 3.1–3.7 from `features_z.parquet`.

**Tests.** `tests/test_tables.py::test_expected_duration_formula` — a hand-built transmat gives literal expected durations. `tests/test_tables.py::test_param_drift_shape` — rows = n_refits × K × d and every (refit_date, state, feature) unique.

**Review evidence.** The first and last transition matrices; `expected_duration.csv` in full; `param_drift.csv` for `dgs10_chg12` and `cpi_3m_ann` pivoted refit_date × state via `to_string()`; a sentence per state on whether its means are stable across refits (question 1 input).

### Step 3.8 — a test that `strategy.py` contains no reference to the smoothed file

**Builds.** `tests/test_no_smoothed_in_strategy.py::test_strategy_never_references_smoothed` — reads the source text of `regime/strategy.py` and asserts the substring `smoothed` (case-insensitive) does not occur, so neither the file name, the config key `outputs_smoothed_probs` nor any function of `run_smoothed_hmm` can be referenced. The strategy is label-source-agnostic: `run.py` section 5 loads label frames (including the smoothed one, for the hindsight rows of the timing grid) and passes them to `strategy` functions as plain `labels` frames. The test passes now against the docstring-only `strategy.py` and constrains section 5.

**Review evidence.** `pytest -q` output.

---

## Section 4 — Conditional statistics

### Step 4.1 — the join of labels to return_{t+1}

**Builds.** `regime/conditional.py`: `join_next_return(labels: pd.DataFrame, factors: pd.DataFrame, cfg: Config) -> pd.DataFrame` — `labels` has index `date` and columns `label, assigned`; `factors` has index `date` (month-end) and the six `cfg.strategy_factors` columns; the result has index `date` = the decision date t, columns `label, assigned` and one column per factor holding the factor return of the next row of the factor index (month t+1), obtained with `factors.shift(-1)` after asserting the factor index is a complete monthly month-end sequence; decision dates without a t+1 return are dropped; restricted to t ≥ `first_window_end`.

**Tests.** `tests/test_conditional.py::test_join_alignment_on_four_rows` — a hand-built four-row example: labels at 2020-01-31 … 2020-04-30, factor returns 2020-01 … 2020-05 with distinct values; the joined return at 2020-01-31 is the 2020-02 return, and so on; the row for the last decision date without a t+1 return is absent (config `sample_first_window_end` replaced to 2020-01-31 for the test).

**Review evidence.** The four-row example printed; `join_next_return(...).head(3)` on the real frames.

### Step 4.2 — `conditional_stats` for all four label sources

**Builds.** `conditional_stats(labels: pd.DataFrame, factors: pd.DataFrame, cfg: Config) -> pd.DataFrame` — over joined rows with `assigned` true, per factor f and hard label k: `n`, `ann_mean = mean × 12`, `ann_std = std(ddof=1) × √12`, `sharpe = mean / std × √12`; columns `factor, state, n, ann_mean, ann_std, sharpe`; `n` is on every row. Section 4 of `run.py` builds the labels frames for the four sources (`hard_labels` of the three probability files; rules with `assigned = True`), writes `outputs/tables/conditional_stats_<source>.csv` with the six columns now and the nine columns after 4.3, and logs the unassigned count per source.

**Tests.** `tests/test_conditional.py::test_conditional_stats_by_hand` — a synthetic joined frame with two states and two factors; literal expected `n, ann_mean, ann_std, sharpe`. `tests/test_conditional.py::test_unassigned_rows_are_excluded_and_counted`.

**Review evidence.** The four tables in full via `to_string()`; unassigned counts per source with the dates.

### Step 4.3 — `block_bootstrap_ci`

**Builds.** `block_bootstrap_ci(labels: pd.DataFrame, factors: pd.DataFrame, cfg: Config) -> pd.DataFrame` — `StationaryBootstrap(cfg.bootstrap_block_size, joined_frame, seed=cfg.run_seed)` on the joined frame (`label, assigned`, six returns) resampling rows jointly; `cfg.bootstrap_n_replications` replications; per replication every conditional Sharpe recomputed; per (factor, state) the `p_low` and `p_high` quantiles; `excludes_zero = (p05 > 0) or (p95 < 0)`. Returns `factor, state, n, ann_mean, ann_std, sharpe, sharpe_p05, sharpe_p95, excludes_zero` and section 4 rewrites the four `conditional_stats_<source>.csv` files with it.

**Tests.** `tests/test_conditional.py::test_bootstrap_reproducible_with_config_seed` — two calls give identical tables. `tests/test_conditional.py::test_excludes_zero_rule` — a hand table of (p05, p95) pairs maps to the expected booleans. `tests/test_conditional.py::test_bootstrap_schema_and_n_unchanged` — `n` equals `conditional_stats`'s `n`; nine columns.

**Review evidence.** The four nine-column tables in full; the count of cells that exclude zero per source (question 2 input); wall-clock.

### Step 4.4 — `filtered_smoothed_gap`

**Builds.** `filtered_smoothed_gap(filt: pd.DataFrame, smooth: pd.DataFrame, factors: pd.DataFrame, cfg: Config) -> pd.DataFrame` — joined frame with both label pairs (`label_f, assigned_f, label_s, assigned_s`) and the six returns on the common out-of-sample dates; same `StationaryBootstrap` settings; per replication and per (factor, state), `gap = sharpe_smoothed − sharpe_filtered`; columns `factor, state, gap, gap_p05, gap_p95`, written to `outputs/tables/filtered_smoothed_gap.csv`.

**Tests.** `tests/test_conditional.py::test_gap_is_zero_when_labels_identical` — identical label frames give `gap == 0` and both percentiles 0.

**Review evidence.** The table in full; the mean absolute gap across cells (with the cells that produced it).

### Step 4.5 — project 1 join if the file exists, else a logged skip

**Builds.** In section 4 of `run.py`: `p1 = load_project1()`; if non-empty, pivot to index `date` × columns `factor`, restrict to the common dates, and run `conditional_stats` + `block_bootstrap_ci` with HMM-filtered labels (with `cfg` replaced so `strategy_factors` is the project 1 factor tuple) into `outputs/tables/conditional_stats_project1.csv`; else log `project1: absent, conditional join skipped`.

**Tests.** `tests/test_project1.py::test_conditional_join_skipped_when_absent` — `caplog` shows the skip line and no file is written to `tmp_path`.

**Review evidence.** The log line; the table if the file existed.

---

## Section 5 — Timed strategy

### Step 5.1 — `trailing_conditional_sharpe` with the strict s < t test

**Builds.** `regime/strategy.py`: `trailing_conditional_sharpe(labels: pd.DataFrame, factors: pd.DataFrame, t: pd.Timestamp, cfg: Config) -> tuple[pd.Series, int]` — with `k = label_t`; over decision dates s < t (strict) with `assigned_s` true and `label_s == k`, using `return_{s+1}` from `join_next_return`, the Sharpe `mean / std(ddof=1) × √12` per factor in `cfg.strategy_universe` (NaN if fewer than 2 observations), and `n_k` = the count of such s. `strategy.py` imports only `pandas`, `numpy`, `regime.config` and `regime.conditional.join_next_return`.

**Tests.** `tests/test_strategy.py::test_trailing_sharpe_is_strictly_before_t` — planting a 1e3 return in month t+1 (and in month t+2) leaves `S` and `n_k` at t unchanged; planting it in month t (earned by decision t−1, which has label k) changes `S`. `tests/test_strategy.py::test_trailing_sharpe_by_hand` — literal expected values on a five-row example.

**Review evidence.** `S` and `n_k` at 2010-12-31 and at `sample_end − 1` for the HMM-filtered labels via `to_string()`.

### Step 5.2 — `weights` with the fallback rules and a test for each fallback branch

**Builds.** `weights(labels: pd.DataFrame, factors: pd.DataFrame, eta: float, cfg: Config) -> pd.DataFrame` — index `date` over the out-of-sample decision dates, columns `cfg.strategy_universe`; at each t: if `assigned_t` is false, or `n_k(t) < cfg.strategy_min_regime_obs`, or `Σ_g S⁺_{g|k}(t) == 0` (including all-NaN), every weight is `1 / len(universe)`; else `w_f = (1 − η) / 5 + η × S⁺_f / Σ_g S⁺_g` with `S⁺ = max(S, 0)` and NaN treated as 0. Rows sum to 1.

**Tests.** `tests/test_strategy.py::test_fallback_unassigned`, `::test_fallback_below_min_regime_obs`, `::test_fallback_no_positive_sharpe` — one synthetic case each, weights exactly 0.2. `tests/test_strategy.py::test_blend_formula_by_hand` — η = 0.5, S = (0.8, −0.2, 0.4, 0.0, NaN) → w = (0.1 + 0.5 × 0.8/1.2, 0.1, 0.1 + 0.5 × 0.4/1.2, 0.1, 0.1) to 1e-12. `tests/test_strategy.py::test_weights_sum_to_one`.

**Review evidence.** Weights at four dates (one per fallback branch if present, one blended) with the `S`, `n_k` behind them; the fraction of months in fallback per source and η.

### Step 5.3 — `backtest` with the literal 3-month test

**Builds.** `backtest(weights: pd.DataFrame, factors: pd.DataFrame, lag: int, cost_bp: float, cfg: Config) -> pd.DataFrame` — pure pandas. Weights dated t earn the factor returns of the row `1 + lag` positions after t in the monthly factor index (t+1 for lag 0, t+2 for lag 1). `gross_ret = Σ_f w_{f,t} r_{f,m}`; `turnover_t = ½ Σ_f |w_{f,t} − w_{f,t−1}|` with `w_{f,t0−1} = 1 / len(universe)` for the first row; `cost = cost_bp / 1e4 × turnover_t`, deducted in the month the weights earn; `net_ret = gross_ret − cost`. Output index `date` = the earning month m, columns `gross_ret, turnover, cost, net_ret`; decision dates whose earning month is beyond the factor data are dropped.

**Tests.** `tests/test_strategy.py::test_backtest_three_month_literal` — this exact example, numbers written literally in the test. Decision dates 2020-01-31, 2020-02-29, 2020-03-31 with weights over (SMB, HML, RMW, CMA, UMD):

```
2020-01-31: 0.2 0.2 0.2 0.2 0.2
2020-02-29: 0.4 0.1 0.1 0.2 0.2
2020-03-31: 0.0 0.0 0.5 0.5 0.0
```

Factor returns:

```
2020-02-29:  0.01  0.02 -0.01  0.00  0.03
2020-03-31: -0.02  0.01  0.04  0.01 -0.01
2020-04-30:  0.03 -0.03  0.02  0.02  0.00
2020-05-31:  0.01  0.01  0.01 -0.02  0.02
```

lag 0, 20 bp:

```
date        gross_ret  turnover  cost    net_ret
2020-02-29   0.0100     0.0      0.0000   0.0100
2020-03-31  -0.0030     0.2      0.0004  -0.0034
2020-04-30   0.0200     0.7      0.0014   0.0186
```

lag 1, 20 bp:

```
date        gross_ret  turnover  cost    net_ret
2020-03-31   0.0060     0.0      0.0000   0.0060
2020-04-30   0.0150     0.2      0.0004   0.0146
2020-05-31  -0.0050     0.7      0.0014  -0.0064
```

All to 1e-12. `tests/test_strategy.py::test_static_weights_have_zero_turnover`.

**Review evidence.** `pytest -q` output; the static backtest's first three and last three rows for lag 1, 20 bp.

### Step 5.4 — the full grid written to `timing_results.csv`

**Builds.** `run_timing_grid(label_frames: dict[str, pd.DataFrame], factors: pd.DataFrame, cfg: Config) -> pd.DataFrame` in `regime/strategy.py`: for every (η ∈ `eta_grid`, lag ∈ `lag_grid`, c ∈ `cost_bp_grid`, source ∈ `label_sources`) — 3 × 2 × 3 × 4 = 72 rows — build the timed weights, the static weights (1/5 on the same dates), backtest both, and record `eta, lag, cost_bp, source, sharpe_static, sharpe_timed, diff, diff_p05, diff_p95, p_one_sided, mean_turnover, n_months` with `sharpe = mean(net_ret) / std(net_ret, ddof=1) × √12` over the common earning months, `diff = sharpe_timed − sharpe_static`, `mean_turnover` of the timed book, and the three interval/p columns NaN until step 5.5. Section 5 of `run.py` loads the four label frames (the smoothed one included, as an ordinary labels frame) and writes `cfg.outputs_timing_results`.

**Tests.** `tests/test_strategy.py::test_timing_grid_has_72_rows_and_unique_keys`. `tests/test_strategy.py::test_static_row_is_identical_across_sources` — `sharpe_static` for a given (lag, c) is the same for all sources and η.

**Review evidence.** `timing_results.csv` in full via `to_string()`; the headline row (η 0.5, lag 1, 20 bp, hmm_filtered) and its static comparator; cumulative net return end values for the headline row.

### Step 5.5 — `timing_gain_bootstrap` filling the interval and p columns

**Builds.** `timing_gain_bootstrap(static: pd.Series, timed: pd.Series, cfg: Config) -> dict` — joint frame of (net_ret_static, net_ret_timed) aligned on the earning month, `StationaryBootstrap(cfg.bootstrap_block_size, frame, seed=cfg.run_seed)`, `cfg.bootstrap_n_replications` replications, per replication `diff = sharpe(timed) − sharpe(static)`; returns `{"diff": point estimate, "p05": ..., "p95": ..., "p_one_sided": fraction of replications with diff ≤ 0}`. Section 5 fills `diff_p05, diff_p95, p_one_sided` for all 72 rows.

**Tests.** `tests/test_strategy.py::test_timing_gain_identical_series` — identical inputs give `diff == 0`, `p05 == p95 == 0`, `p_one_sided == 1.0`. `tests/test_strategy.py::test_timing_gain_reproducible`.

**Review evidence.** The completed `timing_results.csv` in full; the headline row's `diff, diff_p05, diff_p95, p_one_sided` (question 3 input); the lag-0/no-cost versus lag-1/20 bp comparison rows; wall-clock for 72 × 2000 replications.

---

## Section 6 — Robustness

Every robustness run writes under `outputs/regimes/robustness/<name>/` and `outputs/tables/robustness/<name>/` the same files as the main run, and its timing grid to the named file in `outputs/tables/`. Nothing in the main outputs is touched. Each variant is driven by `dataclasses.replace(cfg, ...)`, never by editing `config.toml`.

### Step 6.1 — K ∈ {2, 3, 4, 5} full expanding rerun

**Builds.** `run_robustness_k(cfg: Config) -> None` in `regime/run.py`: for each K in `cfg.hmm_k_grid`, `run_expanding_hmm` and `run_smoothed_hmm` at that K (anchoring, restart tables, state counts, agreement under `robustness/k<K>/`), conditional stats and bootstrap for the sources `hmm_filtered` and `hmm_smoothed`, and the timing grid restricted to those two sources, written to `outputs/tables/timing_results_k<K>.csv`. Fixed here: GMM and rules rows are not rerun because rules do not depend on K and the GMM is the comparison model at `primary_K` only.

**Tests.** `tests/test_robustness.py::test_k_variant_paths_do_not_collide_with_main_outputs` — the variant output paths for each K differ from every main output path.

**Review evidence.** Per K: the headline row of `timing_results_k<K>.csv`, `state_counts` degenerate rows, and the number of anchor disagreements; wall-clock per K (rule 10 applies to the whole step; if exceeded, report and stop).

### Step 6.2 — `covariance_type="diag"` rerun

**Builds.** `run_robustness_diag(cfg)`: `dataclasses.replace(cfg, hmm_covariance_type="diag")` at `primary_K`; `HMMParams.covars` stays (K, d, d) because hmmlearn's `covars_` property returns full-shaped matrices for every covariance type, so `forward_filter` and `anchor` are unchanged; BIC is not recomputed. Outputs under `robustness/diag/` and `outputs/tables/timing_results_diag.csv`.

**Tests.** `tests/test_robustness.py::test_diag_fit_covars_are_diagonal_full_shape` — on synthetic data, the returned `covars` have shape (K, d, d) and zero off-diagonals.

**Review evidence.** Headline row versus the main run's; parameter drift summary for `dgs10_chg12` means.

### Step 6.3 — the 10-feature run from 2004-01-31

**Builds.** `run_robustness_10feat(cfg)`: `dataclasses.replace(cfg, sample_features_from=cfg.sample_robustness_from, sample_first_window_end=cfg.hmm_robustness_first_refit, hmm_covariance_type=cfg.hmm_robustness_covariance_type)` with model-input columns `core + robustness` (10 columns), `primary_K`, first window `[2004-01-31, 2009-12-31]`, refit dates 2009-12-31 and every December after it up to `sample_end`, diagonal covariance (convention 13), outputs under `robustness/10feat/` and `outputs/tables/timing_results_10feat.csv`. The z matrix is not recomputed: the standardised robustness columns already exist in `features_z.parquet` and their standardisation is unchanged (the step 2.3 rule over their non-NaN rows); the run reads them. The out-of-sample window for this variant's conditional statistics and backtests starts at 2009-12-31, since that is its first refit.

**Tests.** `tests/test_robustness.py::test_10feat_model_input_has_ten_columns_from_2004_01_31`. `tests/test_robustness.py::test_10feat_refit_dates_start_2009_12_31`.

**Review evidence.** The first-window restarts table; the headline row; state counts and anchor agreement for the variant.

### Step 6.4 — `min_regime_obs` ∈ {12, 24, 36}

**Builds.** `run_robustness_minobs(cfg)`: for each N in `cfg.strategy_min_regime_obs_grid`, the full 72-row grid with `dataclasses.replace(cfg, strategy_min_regime_obs=N)` and the main run's labels, with bootstrap intervals, to `outputs/tables/timing_results_minobs<N>.csv`.

**Tests.** `tests/test_robustness.py::test_minobs_24_reproduces_main_grid` — the N = 24 file equals `timing_results.csv` exactly.

**Review evidence.** The headline row per N side by side; the fraction of fallback months per N.

### Step 6.5 — `block_size` ∈ {3, 6, 12} for the headline cell only

**Builds.** `run_robustness_blocksize(cfg)`: for each b in `cfg.bootstrap_block_size_grid`, `timing_gain_bootstrap` with `dataclasses.replace(cfg, bootstrap_block_size=b)` for the headline cell (`headline_eta, headline_lag, headline_cost_bp, headline_source`) and the headline cell's conditional Sharpe intervals, written to `outputs/tables/timing_results_blocksize.csv` (columns `block_size` + the `timing_results.csv` columns) and `outputs/tables/conditional_stats_blocksize.csv` (columns `block_size` + the nine conditional columns, `hmm_filtered` only).

**Tests.** `tests/test_robustness.py::test_blocksize_6_reproduces_headline_row`.

**Review evidence.** Both tables in full via `to_string()`.

---

## Section 7 — Outputs and write-up

### Step 7.1 — the three charts

**Builds.** `regime/charts.py`: `regimes_timeline(panel_raw: pd.DataFrame, filt_labels: pd.DataFrame, smooth_labels: pd.DataFrame, param_drift: pd.DataFrame, path: str, cfg: Config)` — two panels, `features_from` to `sample_end`; top: `dgs10_level` and `cpi_3m_ann` (twin axes) with background shading by filtered hard label; bottom: identical with smoothed label; legend entries `"state k: mean dgs10_chg12 = x, mean cpi_3m_ann = y"` from the last refit's `param_drift.csv` rows (two decimals). `conditional_sharpe_heatmap(stats: pd.DataFrame, path: str, cfg: Config)` — factor × state, each cell annotated `"SR / n"` (Sharpe to two decimals, then `n`), a black border on cells with `excludes_zero`; written for `hmm_filtered` and `hmm_smoothed`. `timed_vs_static(backtests: dict[str, pd.DataFrame], path: str, cfg: Config)` — cumulative net return `cumprod(1 + net_ret)` for static, timed-filtered, timed-smoothed at the headline (η, lag, c) on one log-scale axis. Files `outputs/charts/regimes_timeline.png`, `conditional_sharpe_filtered.png`, `conditional_sharpe_smoothed.png`, `timed_vs_static.png` at `cfg.outputs_chart_dpi`. Every `savefig` call in `charts.py` (the step 2.4 chart included) passes `metadata={"Software": None}` so the PNG carries no matplotlib version string.

**Tests.** `tests/test_charts.py::test_three_charts_written_from_synthetic_inputs` — each function writes a non-empty PNG to `tmp_path`. `tests/test_charts.py::test_heatmap_border_only_where_excludes_zero` — inspects the returned `Axes` patches: the count of bordered cells equals the count of `excludes_zero` rows.

**Review evidence.** The four PNGs; the legend strings printed.

### Step 7.2 — all tables and `regime_labels.csv`

**Builds.** `regime/tables.py`: `write_regime_labels(rules, hmm_filt, hmm_smooth, gmm_filt, cfg) -> pd.DataFrame` — index every decision date from `sample_start` to `sample_end`; columns `rules_label, hmm_filtered_label, hmm_filtered_assigned, hmm_smoothed_label, gmm_filtered_label`; labels empty where unavailable, `hmm_filtered_assigned` False where unavailable; written to `cfg.outputs_regime_labels` with `date` as the first CSV column. This is the published series that projects 7 and 9 consume. Every other table named in the kickoff's section 8 is confirmed present with the specified columns by `tables.check_outputs(cfg) -> pd.DataFrame` (columns `path, present, columns_ok`), and section 7 of `run.py` regenerates all of them.

**Tests.** `tests/test_tables.py::test_regime_labels_schema_and_coverage` — synthetic inputs; the five columns, one row per decision date, `assigned` False wherever the label is empty. `tests/test_tables.py::test_check_outputs_lists_every_section_8_table`.

**Review evidence.** `regime_labels.csv` head/tail and value counts per column via `to_string()`; `check_outputs` in full.

### Step 7.3 — `python -m regime.run` end to end from an empty `data/processed/`

**Builds.** Delete everything under `data/processed/` except `.gitkeep` (raw data stays), run `python -m regime.run` (no `--pull`), which runs sections 1 to 7 from the pinned snapshots, appends one row per section to `outputs/tables/runtime.csv` (columns `run_started, section, seconds`), and then run it a second time and confirm the outputs are idempotent: every file under `outputs/` except `outputs/tables/runtime.csv` is compared — CSV files by sha256 of their bytes, PNG files by equality of the decoded pixel array (`matplotlib.image.imread`, `np.array_equal`) — and the table is written to the review file.

**Tests.** `tests/test_run.py::test_run_is_idempotent_for_a_cheap_section` — running section 4 twice into a `tmp_path` copy of the outputs produces identical `conditional_stats_rules.csv` bytes.

**Review evidence.** `runtime.csv` in full; the comparison table (path, compared_as ∈ {bytes, pixels}, sha_run1, sha_run2, identical); total wall-clock; any section over its threshold named.

### Step 7.4 — README answering the four questions

**Builds.** `README.md` replaces "Results pending" with, in order: (1) the number of states chosen with the BIC table and the parameter stability across refits from `param_drift.csv` (one number: the maximum absolute drift of any anchored state mean across refits in z-units, with the refit dates that produced it); (2) whether factor premia differ by regime, with the count of (factor, state) cells whose bootstrap interval excludes 0 out of the total, for filtered and for smoothed; (3) the headline `timing_results.csv` row: `diff`, `[diff_p05, diff_p95]`, `p_one_sided`, before and after costs (the 20 bp row and the 10 bp row; lag 0 and lag 1); (4) the mean `gap` from `filtered_smoothed_gap.csv` and the headline timing `diff` for `hmm_smoothed` versus `hmm_filtered`. Then the three charts (the four PNGs), the headline row verbatim, a "What did not work" section (non-converged restarts, degenerate states, anchor disagreements, dropped rows, any `OPEN.md` items), and a "Limitations" section covering vintage coverage gaps (counts from step 1.4), `n_k` per regime (from the conditional tables), and unmodelled sleeve-internal turnover. Every number in the README is copied from a committed output file and names that file.

**Tests.** `tests/test_readme.py::test_readme_headline_numbers_match_outputs` — parses the headline row quoted in the README and compares it to `timing_results.csv`.

**Review evidence.** The README; for every number in it, the file and row it came from.
