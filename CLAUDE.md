# CLAUDE.md — Macro Regime Classifier and Factor Timing

Sections 0 and 3 of `REGIME_KICKOFF_v2.md`, verbatim. The full build plan is `PLAN.md`; every parameter is in `config.toml`; resolved conventions are in `docs/CONVENTIONS_RESOLVED.md`.

## 0. How this repo is run

Sessions and sections are the same thing after session 0. Session 0 (this session) writes the plan and scaffolding only: no model code, no data pulls, no features. Sessions 1 to 7 each build one complete section of `PLAN.md`, every step in it, in one session. A reviewer (Claude in chat) reads the repo once per section and either clears it or returns fixes. You never start the next section until told to in a new session.

Rules that hold for every session:

1. Do not make design choices. Every parameter, definition, signature, schema, convention and file path is fixed in this document or in `PLAN.md`. If something is genuinely unspecified, append it to `decisions/OPEN.md` with 2 concrete options, finish every other step in the section that does not depend on it, and report it in the review file. Do not pick an option yourself.
2. One commit per step. Message format: `step X.Y: <one line>`. Commit to `main`. Push once at the end of the section. Never force push, never rewrite history, never amend a pushed commit.
3. One review file per section: `review/section_X.md`, following `review/TEMPLATE.md`. Every aggregate number in a review file is backed by raw rows (actual DataFrame rows printed with `to_string()`, never `.describe()` or `.mean()` alone). Anything not verified goes under "Not verified", never silently omitted.
4. `pytest -q` from the repo root must pass at the end of every step. A failing test is reported in the review file with its output; it is never deleted, skipped, marked xfail, or loosened. If a step's test cannot be made to pass, stop the section at that step, complete nothing further, and report.
5. Seeds: `seed = 20260917` in config. Per-restart seeds are `seed + restart_index`. Every call that accepts `random_state` or `seed` receives one from config. No call uses default randomness.
6. Never overwrite anything under `data/raw/`. Every raw pull writes a new file `data/raw/<source>/<series>_<pull_id>.parquet` where `pull_id = YYYYMMDDTHHMMSSZ` at pull time, with a `pulled_at` column. A `data/raw/manifest.csv` (source, series, pull_id, rows, first_date, last_date, sha256) is appended, never rewritten.
7. No lookahead. The rule is in section 3 and enforced by tests. Any code path that uses a value dated after the decision date is a bug, regardless of how small the effect.
8. No library other than those in `pyproject.toml`. If you believe one is needed, it is an `OPEN.md` item.
9. No notebooks. Everything runs from `python -m regime.run` and `pytest`.
10. If the runtime of a step exceeds 20 minutes on your machine, stop and report the timing in the review file rather than reducing restarts, replications or the grid.

## 3. Timing convention

Decision date t is the last calendar day of month t. The decision made at t is executed at the close of the first trading day of t+1 and earns the factor return of month t+1 (no-lag case) or t+2 (1-month-lag case).

Information set at t:

- Market series (DGS10, DGS2, DTWEXBGS, DTWEXM, DCOILWTICO, VIXCLS, T10YIE): the last non-missing daily observation dated on or before t, looking back at most 10 calendar days. If no observation exists within 10 days the month is NaN and reported. These series are not revised.
- Revised macro series (CPIAUCSL, INDPRO, UNRATE): the ALFRED vintage as of t. For each observation month m, the value used is the one from the release with the greatest `realtime_start` that is ≤ t. All lags of a revised series at decision date t (m−3, m−12) come from that same as-of-t vintage, never from the latest vintage. In practice CPI and INDPRO at t refer to month t−1 and UNRATE to month t−1.
- Factor returns: months ≤ t.

Lags are in decision-date index units: "12 months ago" means 12 rows earlier in the decision-date index, not calendar arithmetic on days.

Everything downstream reads one wide monthly DataFrame produced by `regime/data/asof.py` (schema in section 10). Nothing downstream touches raw series.
