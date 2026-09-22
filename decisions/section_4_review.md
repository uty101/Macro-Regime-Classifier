# Decision — the reviewer's answers to the section 4 questions

The four questions are those in `instructions/04_section_4.status.md`,
"Questions and blockers for the reviewer". The reviewer answered them in the
session 5 instruction message (`instructions/05_section_5.md`); this file
records the answers so nothing in section 5 onward has to go back to chat for
them.

## Q1 — section 5 builds a timed strategy on a classifier that does not separate the factor premia

**Section 5 proceeds, and the write-up is framed around the null.**

`hmm_filtered` produces 1 of 18 pairwise state differences whose interval
excludes zero, below the ~2 of 18 a 90% interval gives by chance. Section 5
still runs every step of the plan. What changes is the framing: the README and
every review and status file are written around **the null that section 4
found**, not around whichever cell of the 72-row timing grid happens to come
out positive. The headline cell is fixed in advance — η 0.5, lag 1, 20 bp,
`hmm_filtered` (`strategy.headline_*` in `config.toml`) — and is the only cell
any summary may lead with. The other 71 rows are a grid, reported as a grid,
never as a result.

A timing gain near zero, or negative once costs are paid, is the expected
outcome of this project and is reported as such.

## Q2 — should the refit split be bootstrapped?

**Yes. The refit split gets intervals.**

`conditional_stats_refit_split.csv` (`hmm_filtered` only) gains `sharpe_p05`,
`sharpe_p95` and `excludes_zero` per (factor, state, `run_started_on_refit`),
and the difference between the two flags per (factor, state) — with its own
`diff_p05`, `diff_p95` and `excludes_zero` — goes to
`outputs/tables/conditional_refit_split_differences.csv`.

Same `StationaryBootstrap` settings and the same seed as step 4.3, and **both
flags are resampled in one draw**, so each replication's difference is a
like-for-like comparison on one resampled history rather than a difference
between two independently resampled marginals. This is the same construction
as the pairwise state differences and the filtered/smoothed gap.

RMW state 0 splitting 3.228 (n = 22) against 0.263 (n = 69) is what prompted
this; with intervals the table can say whether that split is more than noise
instead of merely reporting that it is large.

## Q3 — should the conditional tables report the excess over unconditional?

**Yes. Report excess over unconditional.**

Every `conditional_stats_<source>.csv` gains
`excess_sharpe = sharpe − the factor's unconditional Sharpe`, taken from
`outputs/tables/unconditional_stats.csv` over the identical out-of-sample
dates. Mkt-RF paying 0.668 unconditionally is the reason: a conditional 0.712
is a level, not a finding, and the two must be readable side by side.

Every `conditional_differences_<source>.csv` **keeps its columns unchanged**. A
difference of two conditional Sharpes already nets the unconditional level out,
so subtracting it again would be subtracting zero and adding a column that says
nothing.

Step 7.4 reports `excess_sharpe` beside `sharpe`, and the refit-split
differences beside the pairwise ones.

## Q4 — the deviations from the letter of the specification

**All accepted.** The seven extra tests, the one-line change to
`tests/test_run.py::test_unbuilt_section_raises`, and `bootstrap_conditional`
returning three frames with `block_bootstrap_ci` as a thin wrapper over it are
all additive, all flagged, and all already reflected in `PLAN.md`.

## What this file does not do

It does not change a convention, a `config.toml` value, or any section 1 to 3
output. Sections 1 to 4 are re-run with the addendum only so the committed
tables carry the new columns; the numbers that were already there do not move.
