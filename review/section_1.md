# Review — Section 1: Foundation and data

## Section

This is Section 1 of `PLAN.md` (steps 1.1 to 1.7): the package skeleton and `Config`, the `FredClient` raw-write conventions, the FRED market pulls and `month_end_market`, the ALFRED vintage pulls and `build_asof`, the French pull, the project 1 adapter, and the as-of panel `data/processed/asof_panel.parquet`. **The section stopped before step 1.3.** The session instruction was to confirm `FRED_API_KEY` is set before starting and, if it is not, to stop before step 1.3 and report here. It is not set: it is absent from the process environment, the User scope and the Machine scope (`[Environment]::GetEnvironmentVariable("FRED_API_KEY", "User")` and `("FRED_API_KEY", "Machine")` are both `$null`), and there is no `.env` in the repo. Steps 1.1 and 1.2 (no network) are complete and committed; steps 1.3 to 1.7 were not started. Step 1.6 (project 1 adapter) needs no network, but it was held with the rest because the instruction was to stop before 1.3, not to skip 1.3, so the step order and the commit order stay as `PLAN.md` lists them. Nothing was pushed: the instruction was to push once after step 1.7.

To resume: set `FRED_API_KEY` in the environment and start a new session at step 1.3.

## Steps completed

- `step 1.1: package skeleton, Config with load_config, section registry and the key round-trip test` — `ff81cbf`
- `step 1.2: FredClient with one pull_id per session, write_raw that never overwrites, and the append-only manifest` — `1fb67ac`

Not started: 1.3, 1.4, 1.5, 1.6, 1.7.

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

Every `config.toml` array is a `tuple`; `hmm_tol`, `gmm_tol`, `hmm_assigned_threshold`, `hmm_anchor_tie_tolerance`, `bootstrap_p_low`, `bootstrap_p_high` and `strategy_headline_eta` are `float`; the three `*_pull_id` fields and `sample_end` are the empty `str` placeholders (they are filled at steps 1.3, 1.4 and 1.5, which did not run).

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

## Tests run

Final state of the repo (after the step 1.2 commit), from the repo root. `.venv` was created with `uv venv --python 3.11` and `uv pip install -e .` from `pyproject.toml`; no library outside `pyproject.toml` was installed and `uv.lock` was not created.

```
$ pytest -q
.........                                                                [100%]
9 passed in 0.53s
```

The nine tests: `tests/test_config.py::test_config_keys_round_trip`, `tests/test_config.py::test_config_is_frozen_and_typed`, `tests/test_run.py::test_sections_registered_in_order`, `tests/test_run.py::test_unbuilt_section_raises`, `tests/test_fred_client.py::test_missing_api_key_raises`, `tests/test_fred_client.py::test_pull_id_format`, `tests/test_fred_client.py::test_write_raw_never_overwrites`, `tests/test_fred_client.py::test_manifest_appends`, `tests/test_fred_client.py::test_sha256_of_overrides_parquet_digest`. `test_unbuilt_section_raises` and `test_sha256_of_overrides_parquet_digest` are additional to the ones `PLAN.md` names; every named one is present. No test touches the network.

## Runtime per step

| step | wall-clock | machine | notes |
|---|---|---|---|
| env | ~3:00 | Windows 11 Home, 16 GB, Python 3.11.15 via uv | `.venv` creation and `uv pip install -e .`, before step 1.1 |
| 1.1 | 1:07 | same | 14:13:22Z to 14:14:29Z |
| 1.2 | 1:40 | same | 14:14:29Z to 14:16:09Z |
| 1.3 to 1.7 | — | | not started: `FRED_API_KEY` not set |

No step approached the 20-minute threshold (`run.step_timeout_minutes`).

## Not verified

- `FredClient` against the live FRED API: constructing `fredapi.Fred` makes no request, so `test_pull_id_format` passes with a dummy key; nothing proves the key or the endpoint.
- `pull_market`, `pull_vintages`, `month_end_market`, `build_asof`, `pull_french`, `load_project1`, `build_asof_panel`: not built.
- `fred.market_pull_id`, `fred.vintage_pull_id`, `french.pull_id`, `sample.end`: still `""`; the two-phase assertion in `tests/test_config.py` is at its loose phase (`str` only) for all four.
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

This review: added `review/section_1.md`.

Not committed and ignored: `.venv/` (`.gitignore`). Untracked and untouched: `Project Outline/` (present before this session).

## Reviewer reads

1. `regime/config.py` — the 86 field names against `config.toml`, the `float` coercion of int-valued TOML numbers, and that `KeyError` names the key in both directions.
2. `tests/test_config.py` — the two-phase comment on the `*_pull_id` / `sample_end` assertion; it is at the loose phase and steps 1.3 to 1.5 must each tighten one key in the same commit.
3. `regime/data/fred.py` — `pull_id` stamped once in `__init__`; `write_raw` raises `FileExistsError` before touching anything; the manifest is opened in append mode only; `sha256_of` overrides the parquet digest.
4. `regime/run.py` — the `SECTIONS` registry and that `--pull` reaches only section 1.
5. The first paragraph of this file — why the section stopped and what is needed to resume.
