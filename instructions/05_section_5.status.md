# Status — 05, section 5

**Outcome: completed.** The section 4 addendum and every step of Section 5 of
`PLAN.md` (5.1 to 5.5) were built, in one session, without pausing between
steps. The session did not stop early, did not refuse anything under rule 1,
and hit no rule 4 or rule 10 threshold.

## Step reached

All of 5.1 to 5.5, plus the two commits the instruction message required before
them.

| step | what | commit |
|---|---|---|
| — | this instruction message, saved verbatim (rule 11) | `15ea288` |
| — | the section 4 addendum and the reviewer's four answers | `4b0258d` |
| 5.1 | `trailing_conditional_sharpe`, over `s < t` strictly | `a278dc9` |
| 5.2 | `weights`, its three fallbacks, `static_weights` | `a2f56e4` |
| 5.3 | `backtest`, indexed by the earning month | `f355224` |
| 5.4 | `run_timing_grid`, section 5 of `run.py`, the three evidence tables | `a1279ca` |
| 5.5 | `timing_gain_bootstrap` and `fill_timing_gain` | `8d3ef4b` |
| — | `review/section_5.md` | `8bffc5b` |

`pytest -q`: **101 passed**, at the end of every step and at the end of the
section.

## The headline cell

The cell fixed in `config.toml` before any of this ran — η 0.5, lag 1, 20 bp,
`hmm_filtered`:

```
 eta  lag  cost_bp       source  sharpe_static  sharpe_timed   diff  diff_p05  diff_p95  p_one_sided  mean_turnover  n_months
 0.5    1       20 hmm_filtered       0.206378      0.179378 -0.027 -0.227855  0.170963       0.5865       0.043992       257
```

**Timing the five Fama-French sleeves on the filtered HMM's regime labels did
not beat holding them equally weighted.** The point estimate is slightly
negative, the 90% interval spans zero widely, and `p_one_sided = 0.5865` — the
bootstrap's replications split almost exactly evenly on the sign. This is the
expected outcome. Section 4 found the same classifier separating 1 of 18
pairwise state premia, below chance; a strategy built on a signal that weak was
not going to pay for its own trading.

The headline diff is **not** positive with `p_one_sided` below 0.1, so the
confirmation check the instruction message specified was not triggered. The
three conditions it names are nonetheless all satisfied and were verified:

- **The weights at t use only `s < t`.** `trailing_conditional_sharpe` selects
  `joined.index < t`, strictly, and
  `tests/test_strategy.py::test_trailing_sharpe_is_strictly_before_t` plants a
  1e3 return in month t+1 and in month t+2 and asserts `S` and `n_k` do not
  move, then plants it in month t — earned by the decision at t−1, which is
  `< t` — and asserts they do.
- **The lag is applied as specified.** Weights dated t earn the factor row
  `1 + lag` positions later, positionally, on an index `join_next_return` has
  already asserted to be a complete monthly month-end sequence.
  `test_backtest_three_month_literal` checks PLAN.md step 5.3's example at both
  lags with every number written out.
- **The static comparator runs on identical earning months.** `timing_cells`
  asserts `timed.index.equals(static.index)` for each of the 72 cells and
  raises otherwise; `test_static_row_is_identical_across_sources` checks that
  `sharpe_static` is one number per (lag, cost_bp) across all sources and η.

## The grid, as a grid

The other 71 rows are in `outputs/tables/timing_results.csv` in full. Counts
over all 72:

```
rows: 72
p_one_sided < 0.1: 0
diff > 0: 20
interval excludes zero: 7
```

**No cell of the grid has `p_one_sided` below 0.1.** 20 cells have a positive
point estimate; none of the 20 is distinguishable from zero. The 7 cells whose
interval excludes zero are all `rules` and all negative — the backward-looking
quadrants destroy Sharpe, more so as η and the cost rise.

The hindsight `hmm_smoothed` labelling gives the grid's largest positive point
estimate, +0.0726 at η 0.5 / lag 0 / 10 bp, at `p_one_sided = 0.249`. Perfect
regime knowledge, with the answers supplied in advance, still does not produce
a timing gain this test can tell from zero.

## Three things the review file puts beside the number

1. **The tilt was taken.** `weight_deviation.csv`: `hmm_filtered` leaves 1/5 on
   186 of 258 months, mean absolute deviation 0.078 at η 0.5, maximum 0.4 (a
   weight of 0.6). The null is not "the strategy never traded".
2. **The lag and the costs are each larger than the gross signal.**
   `timing_headline_decomposition.csv`: lag 0 / 0 bp gives +0.0185; 20 bp takes
   0.0221 of it and the one-month lag takes 0.0232. Neither alone explains the
   result — the signal is too small to survive either.
3. **The headline flips sign if the last month is removed.** Dropping
   2026-07-31, the single largest timed-minus-static month of 257, moves
   `diff` from −0.027 to +0.013. Reported in full in the review file. It cuts
   both ways: a number one month can flip is a null, not a negative finding.

## The section 4 addendum

All four reviewer answers implemented and recorded in
`decisions/section_4_review.md`; `PLAN.md` steps 4.2, 4.3 and 7.4 amended;
sections 1 to 4 re-run with no `--pull` and the regenerated tables committed.
The section 4 numbers already reported did not move.

- **Q2, intervals on the refit split.** `bootstrap_refit_split` carries
  `run_started_on_refit` as a column of the resampled array beside the label
  and the six returns, so both halves come from one draw. **0 of 18
  refit-split differences exclude zero.** The splits that prompted the question
  — RMW state 0 at 3.228 against 0.263 — are noise at these sample sizes. Ten
  of the 36 individual halves have their own Sharpe excluding zero, but no pair
  of halves differs.
- **Q3, `excess_sharpe`.** Every `conditional_stats_<source>.csv` now carries
  it beside `sharpe`. It does what the reviewer expected: `hmm_filtered` Mkt-RF
  state 2 reads 0.712 and excludes zero, but the pooled Mkt-RF Sharpe over the
  same 258 dates is 0.6676, so the excess is +0.0445.
- **Q1 and Q4** are framing and acceptance; recorded in the decisions file.

## Rule 10 — runtime

No step came close to its threshold. Nothing was reduced:
`bootstrap.n_replications` is the configured 2000, `bootstrap.block_size` the
configured 6, and all 72 cells were run.

```
step                          seconds
addendum (sections 1-4 rerun)      24
5.1 + 5.2 (weights, all cells)      2
5.3 + 5.4 (72-cell grid)            3
5.5 (72 cells x 2000 reps)         10
section 5 end to end               24
```

Step 5.5's threshold is 60 minutes; it took 10 seconds.

## Deviations from the letter of the specification

All additive, all flagged here rather than left to be found.

- **Seven tests beyond those `PLAN.md` and the instruction message list:**
  `test_trailing_sharpe_excludes_unassigned_and_other_states`,
  `test_backtest_drops_decision_dates_without_an_earning_month`,
  `test_timing_gain_is_signed_timed_minus_static`,
  `test_fill_timing_gain_completes_every_row`, and in the addendum the
  `KeyError` branch of `test_excess_sharpe_is_conditional_minus_unconditional`
  plus the run-structure assertion inside
  `test_refit_split_difference_zero_when_flags_identical`. The signed-gain test
  exists because every other step 5.5 test asserts a zero, and a bootstrap that
  always returned zero would pass all of them.
- **`weights` and `backtest` attach one entry each to `DataFrame.attrs`** —
  the fallback branch per row and the decision date per earning month — both as
  tuples. `fallback_branches(book)` rebuilds the Series. They are tuples rather
  than Series because pandas compares `attrs` values with `==` when it merges
  two frames' metadata, and a Series there raises "truth value is ambiguous"
  the first time a backtest frame is concatenated; this was found by a failing
  test in step 5.5 and fixed there.
- **Three tables beyond the plan's outputs**, all named in the instruction
  message: `weight_deviation.csv`, `timing_headline_decomposition.csv`,
  `headline_turnover_top10.csv`. The decomposition table exists because
  `strategy.cost_bp_grid` starts at 10 bp, so the grid alone cannot separate
  the cost of the lag from the cost of trading.
- **`timing_cells` is an extra function** that `run_timing_grid` and
  `fill_timing_gain` share, so the 72 cells are built once and step 5.5's
  intervals sit on exactly the series step 5.4's point estimates came from.
  `fill_timing_gain` re-checks each cell's `diff` against the grid's and raises
  on disagreement; all 72 agreed to 1e-12.
- **`tests/test_run.py::test_unbuilt_section_raises`** asserted that section 5
  was not built; it now asserts it of section 6. One line, in the step 5.4
  commit.
- **`regime/strategy.py`'s module docstring cannot name the step 3.8 test by
  filename**, because that filename contains the very substring the test
  forbids. It describes the test instead.

## Questions and blockers for the reviewer

Nothing blocked the work and nothing was appended to `decisions/OPEN.md`.
Three things want a decision before section 6, in descending order of
importance:

1. **The project's result is now a null in three places, and the README should
   say so plainly.** Section 3 found the filtered labels moving mostly at
   refits; section 4 found 1 of 18 pairwise state differences excluding zero,
   below chance; section 5 finds 0 of 72 timing cells with `p_one_sided` below
   0.1, including every cell of the hindsight labelling. Per the Q1 answer
   already given, step 7.4 will be framed around that. The reviewer may want to
   go further and have the README state the null in its first paragraph rather
   than after the four questions are worked through. That is a step 7.4
   instruction, not a session choice.

2. **The 24-month `min_regime_obs` holds 28% of the out-of-sample window at
   1/5, and step 6.4 is the only place that is tested.** The 72 fallback months
   split exactly 24/24/24 across the three states — state 1 did not reach its
   24th observation until 2023-03-31, eighteen years into the window. That is
   a large share of the sample in which the strategy is definitionally the
   comparator, and it mechanically shrinks any `diff` toward zero. Step 6.4
   sweeps `min_regime_obs` over {12, 24, 36} as planned; the reviewer should
   know before it runs that 12 will produce a larger `diff` in either
   direction, and that this is a mechanical effect rather than evidence.

3. **The headline flips sign on one month out of 257, and nothing in the plan
   catches that class of fragility.** Section 4 hit the same thing from the
   other side: its one `hmm_filtered` pairwise difference leaned on 2009-03.
   There is no leave-one-month-out or trimmed statistic anywhere in `PLAN.md`,
   and step 6.5 sweeps only `block_size`. If the reviewer wants one, it is a
   PLAN change and belongs in section 6, not a session choice.

## Final remote state

```
$ git log origin/main --oneline -3
8bffc5b section 5: review file
8d3ef4b step 5.5: the timing-gain bootstrap, filling the interval and p columns
a1279ca step 5.4: the 72-cell timing grid, and what the weights actually do
```

That log is as of the review-file commit. This status file is committed and
pushed on top of it, so on the remote the head is one commit later:
`05: status file`.
