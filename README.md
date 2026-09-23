# Macro Regime Classifier and Factor Timing

**The result is a null, and it is the point of the project.** The filtered
classifier's labels move at refits as often as not — 9 of its 20 label changes
land exactly on a refit date, 45% against a 1-in-12 base rate
(`outputs/tables/classifier_diagnostics.csv`, `core` row). It separates factor
premia no better than chance: 1 of 18 pairwise state differences has a
bootstrap interval excluding zero
(`outputs/tables/conditional_differences_hmm_filtered.csv`), where a 90%
interval on 18 independent draws would give about 2. And no cell of the
72-row timing grid is distinguishable from zero in the direction of a gain —
0 of 72 cells reach `p_one_sided` below 0.10, the smallest being 0.178
(`outputs/tables/timing_results.csv`). The headline cell, fixed in
`config.toml` before any of it was computed, is a Sharpe difference of
**−0.027 with a 90% interval of [−0.228, 0.171]** and `p_one_sided` 0.5865:
indistinguishable from zero in both directions, not a negative result.

A reader who stops here has the finding. Everything below is the evidence.

Data from FRED (Federal Reserve Bank of St. Louis), redistributed unmodified under its terms.
Vintage data from ALFRED (Federal Reserve Bank of St. Louis), redistributed unmodified under its terms.
Factor returns from the Kenneth R. French Data Library, redistributed unmodified under its terms.

---

## What this is and how to run it

A monthly macro regime classifier built from US rates, inflation, growth, the
dollar, oil and equity volatility, estimated so that the regime at month-end t
uses only data published by month-end t — ALFRED vintages for the revised
series, a 10-day lookback for the market series.

Regimes come from an expanding-window Gaussian HMM (primary), a Gaussian
mixture and a growth/inflation rules quadrant, refitted every 12 months and
anchored so that state numbers mean the same thing across refits. They
condition monthly Fama-French factor returns with stationary-block-bootstrap
intervals, and drive a timed allocation over SMB, HML, RMW, CMA and UMD tested
out of sample after a 1-month implementation lag and 20 bp of costs.

```bash
python -m regime.run          # sections 1 to 7, from the pinned raw snapshots
pytest -q                     # 129 tests
```

**The repo's raw data snapshots are pinned and committed.** `data/raw/` holds
the FRED and ALFRED parquet pulls, the French zips and `manifest.csv` in full
(convention 14), and `config.toml` pins the `pull_id` of each. A fresh clone
therefore reproduces every table and every chart in this file **without a FRED
API key** and without touching the network; `pytest` passes the same way.
`data/interim/`, `data/processed/` and `data/external/` are regenerated from
the raw snapshots on every run and are never committed.

The plan is `PLAN.md`, the rules `CLAUDE.md`, the resolved conventions
`docs/CONVENTIONS_RESOLVED.md`, and every parameter `config.toml`. Every number
in this file is copied from a committed output file and names the file it came
from.

---

## The four questions

### 1. How many persistent states are there, and are they stable across refits?

**Three.** `primary_K` is the argmin of BIC over `hmm.k_candidates` = {3, 4},
fitted on the 168-row first window (`outputs/tables/bic_by_k.csv`):

```
 K       loglik   m   T         bic  converged
 2 -1382.020801  90 168 3225.198361       True
 3 -1095.357855 138 168 2897.822739       True
 4  -989.897103 188 168 2943.099433       True
 5  -918.452448 240 168 3066.656250       True
```

K = 3 wins on the full grid as well as on the candidate pair, by 45 BIC points
over K = 4.

**They are not stable.** The largest absolute drift of any anchored state mean
between consecutive refits is **1.88 z**: state 2's `indpro_chg12` mean moves
from −2.329 at the 2013-12-31 refit to −0.451 at the 2014-12-31 refit. Both
rows are in `outputs/tables/param_drift.csv`:

```
refit_date  state      feature      mean  variance
2013-12-31      2 indpro_chg12 -2.328620  1.633968
2014-12-31      2 indpro_chg12 -0.451311  2.071638
```

A state whose mean on one of eight features moves nearly two standard
deviations in one refit is not the same state in any economic sense, and this
is the mechanism behind the 45% of label changes that land on refit dates. The
per-refit matching distances are in `outputs/tables/anchor_chain.csv`: 18 of
66 state-refit rows were matched at more than 1 z, the largest at 2.689.

### 2. Do factor premia differ by regime by more than noise?

**No, under the filtered labels.** The question is about *differences* between
states, so it is answered with
`outputs/tables/conditional_differences_hmm_filtered.csv` and not with the
per-cell table: a conditional Sharpe that excludes zero in one state says only
that the factor pays there, which it may do in every state.

One of 18 pairwise differences excludes zero:

```
factor  state_a  state_b  sharpe_diff  diff_p05  diff_p95  excludes_zero  n_a  n_b
   UMD        0        2    -1.024094 -1.740244 -0.112286           True   91  106
```

Under smoothed labels, three of 18 do
(`outputs/tables/conditional_differences_hmm_smoothed.csv`):

```
factor  state_a  state_b  sharpe_diff  diff_p05  diff_p95  excludes_zero  n_a  n_b
Mkt-RF        0        1     1.238306  0.350368  2.179252           True   73   38
   HML        0        2     0.901747  0.146182  1.666262           True   73  147
   RMW        0        2     0.908286  0.143180  1.587192           True   73  147
```

The per-cell table is `outputs/tables/conditional_stats_hmm_filtered.csv`,
where `excess_sharpe` sits beside `sharpe` so a level is not read as a finding:

```
factor  state   n  ann_mean  ann_std    sharpe  excess_sharpe  sharpe_p05  sharpe_p95  excludes_zero
Mkt-RF      0  91  0.048303 0.140195  0.344543      -0.323032   -0.406935    1.344566          False
Mkt-RF      1  61  0.146656 0.134906  1.087095       0.419520    0.588614    1.773611           True
Mkt-RF      2 106  0.122672 0.172274  0.712075       0.044500    0.174042    1.282782           True
   SMB      0  91 -0.012976 0.086864 -0.149381      -0.115147   -0.712789    0.421516          False
   SMB      1  61 -0.027639 0.104311 -0.264971      -0.230737   -0.766503    0.364228          False
   SMB      2 106  0.019381 0.089124  0.217462       0.251697   -0.347618    0.787990          False
   HML      0  91 -0.039890 0.086716 -0.460008      -0.431327   -1.054890    0.177871          False
   HML      1  61  0.026990 0.106786  0.252751       0.281431   -0.395023    1.043652          False
   HML      2 106  0.011004 0.129359  0.085064       0.113744   -0.595287    0.701788          False
   RMW      0  91  0.048541 0.051418  0.944037       0.496902    0.287502    1.646790           True
   RMW      1  61 -0.000944 0.099138 -0.009525      -0.456660   -0.698222    0.666484          False
   RMW      2 106  0.037947 0.070457  0.538587       0.091452   -0.067646    1.053632          False
   CMA      0  91 -0.018145 0.050791 -0.357253      -0.398462   -0.917908    0.158767          False
   CMA      1  61 -0.027718 0.087694 -0.316075      -0.357285   -1.012381    0.391057          False
   CMA      2 106  0.038355 0.067444  0.568694       0.527485   -0.175451    1.144325          False
   UMD      0  91  0.094378 0.124003  0.761095       0.645527    0.306703    1.246084           True
   UMD      1  61  0.016584 0.139121  0.119203       0.003636   -0.452916    0.755164          False
   UMD      2 106 -0.047434 0.180358 -0.262999      -0.378566   -0.814666    0.544030          False
```

Mkt-RF state 1's 1.087 is the clearest example of why the per-cell count is
the wrong statistic: the pooled Mkt-RF Sharpe over the identical dates is
0.668 (`outputs/tables/unconditional_stats.csv`), so the excess is 0.420, and
the state-1-against-state-0 difference of 0.743 does not exclude zero.

```
factor   n  ann_mean  ann_std    sharpe  sharpe_p05  sharpe_p95
Mkt-RF 258  0.102112 0.152959  0.667575    0.258644    1.096812
   SMB 258 -0.003149 0.091978 -0.034235   -0.357822    0.301642
   HML 258 -0.003167 0.110440 -0.028680   -0.433182    0.376164
   RMW 258  0.032488 0.072659  0.447135    0.092335    0.816471
   CMA 258  0.002805 0.068059  0.041209   -0.358389    0.403102
   UMD 258  0.017721 0.153339  0.115567   -0.244422    0.574285
```

Beside them sits the refit split. `outputs/tables/conditional_stats_refit_split.csv`
splits the out-of-sample dates by whether the filtered run containing them
began on a refit date, and
`outputs/tables/conditional_refit_split_differences.csv` bootstraps the
difference between the two halves in one draw. **Six of 18 exclude zero** —
six times as many as the pairwise state differences do:

```
factor  state  sharpe_diff  diff_p05  diff_p95  excludes_zero  n_false  n_true
   HML      1     1.911315  0.242195  4.345503           True       49      12
   HML      2    -1.776864 -3.278472 -0.579432           True       23      83
   RMW      0    -2.965494 -5.865135 -2.139766           True       22      69
   RMW      2    -1.383588 -2.686924 -0.191373           True       23      83
   CMA      1     2.565516  1.207959  3.950816           True       49      12
   UMD      0    -0.986036 -2.441870 -0.099709           True       22      69
```

Whether a month's label came from a fresh refit separates the conditional
Sharpes more reliably than which state the label names. That is a statement
about the estimator, not about the macroeconomy, and it is the most
uncomfortable number in the project. The `n_false`/`n_true` columns say why it
should not be over-read: several of these cells rest on 12 to 23 months.

### 3. Does timing beat static equal weight out of sample, after lag and costs?

**No, and it does not lose either.** The headline cell — η 0.5, lag 1, 20 bp,
`hmm_filtered`, fixed in `config.toml` before section 5 ran — from
`outputs/tables/timing_results.csv`:

```
 eta  lag  cost_bp       source  sharpe_static  sharpe_timed   diff  diff_p05  diff_p95  p_one_sided  mean_turnover  n_months
 0.5    1       20 hmm_filtered       0.206378      0.179378 -0.027 -0.227855  0.170963       0.5865       0.043992       257
```

The same cell at 10 bp, and both at lag 0 (same table):

```
 eta  lag  cost_bp       source  sharpe_static  sharpe_timed      diff  diff_p05  diff_p95  p_one_sided  mean_turnover  n_months
 0.5    0       10 hmm_filtered       0.211710      0.219170  0.007460 -0.164905  0.165940       0.4655       0.043875       258
 0.5    0       20 hmm_filtered       0.211710      0.208130 -0.003580 -0.176491  0.155384       0.5115       0.043875       258
 0.5    1       10 hmm_filtered       0.206378      0.190541 -0.015837 -0.216215  0.181498       0.5510       0.043992       257
 0.5    1       20 hmm_filtered       0.206378      0.179378 -0.027000 -0.227855  0.170963       0.5865       0.043992       257
```

Halving the cost moves the headline `diff` from −0.027 to −0.016; removing the
lag moves it to −0.004. Both stay inside an interval two-thirds of a Sharpe
point wide. The gross signal is in
`outputs/tables/timing_headline_decomposition.csv`, which holds the source and
η fixed and varies only the lag and the cost, because the grid has no
zero-cost column:

```
                label  eta  lag  cost_bp       source  sharpe_static  sharpe_timed      diff
             headline  0.5    1       20 hmm_filtered       0.206378      0.179378 -0.027000
      no lag, no cost  0.5    0        0 hmm_filtered       0.211710      0.230209  0.018499
no lag, headline cost  0.5    0       20 hmm_filtered       0.211710      0.208130 -0.003580
headline lag, no cost  0.5    1        0 hmm_filtered       0.206378      0.201703 -0.004675
```

Before lag and costs the timing signal is worth +0.018499 Sharpe. Applying the
lag alone takes it to −0.004675; applying the cost alone takes it to −0.003580;
applying both gives the headline −0.027000. Neither the lag nor the cost is the
whole story — each on its own is enough to remove the gross signal. This is the
outcome the project's own outline predicted in advance: "timing adds 0.1 to 0.2
Sharpe before costs and roughly zero after". It did not reach 0.1 before costs.

### 4. How much of the benefit is hindsight?

**Hindsight helps the classifier, and still does not produce a tradeable
gain.** Both halves are true and neither on its own is the answer.

*Hindsight helps.* Smoothed labels give 3 of 18 pairwise state differences
excluding zero against the filtered labels' 1 (the two tables quoted under
question 2), and every smoothed timing cell at η 0.5 is positive where the
filtered ones are negative (`outputs/tables/timing_results.csv`):

```
 eta  lag  cost_bp       source  sharpe_static  sharpe_timed     diff  diff_p05  diff_p95  p_one_sided  mean_turnover  n_months
 0.5    0       10 hmm_smoothed       0.211710      0.284289 0.072579 -0.103844  0.251723       0.2490       0.035955       258
 0.5    0       20 hmm_smoothed       0.211710      0.275858 0.064149 -0.110806  0.243513       0.2760       0.035955       258
 0.5    1       10 hmm_smoothed       0.206378      0.259884 0.053507 -0.145811  0.253078       0.3355       0.036095       257
 0.5    1       20 hmm_smoothed       0.206378      0.251173 0.044795 -0.154583  0.243486       0.3625       0.036095       257
```

*And it is not enough.* The headline-equivalent smoothed cell is +0.045 with
an interval of [−0.155, 0.243] and `p_one_sided` 0.3625. The best cell in the
entire 72-row grid is the smoothed +0.073 at lag 0 and 10 bp, and its
`p_one_sided` is 0.2490. **A labelling that is allowed to see the whole
sample, traded with no lag and at half the headline cost, still cannot clear
zero.** Since the smoothed labelling is not investable at all — it uses the
future — this is a ceiling, not a strategy.

The cell-level gap tells the same story from the other side.
`outputs/tables/filtered_smoothed_gap.csv` reports smoothed minus filtered per
factor and state, bootstrapped in one draw so each replication compares the
two labellings on one resampled history. The **mean gap over its 18 cells is
−0.034**, and only one cell excludes zero:

```
factor  state       gap   gap_p05   gap_p95
   RMW      0 -1.065696 -1.900999 -0.294969
```

So hindsight does not lift the conditional Sharpes cell by cell; it buys a
cleaner *separation* between states, which is what a timing rule consumes.
Filtered and smoothed labels agree on 68.2% of dates
(`outputs/tables/classifier_diagnostics.csv`, `core` row,
`filtered_smoothed_agreement`), and the two panels of
`outputs/charts/regimes_timeline.png` are that 31.8% drawn.

---

## The charts

| file | what it shows |
|---|---|
| `outputs/charts/regimes_timeline.png` | `dgs10_level` and `cpi_3m_ann`, 1991-01-31 to 2026-07-31, shaded by filtered label (top) and by smoothed label (bottom). The difference between the panels is question 4. |
| `outputs/charts/conditional_sharpe_filtered.png` | factor × state conditional Sharpe under filtered labels, each cell `SR / n`, black border where the interval excludes zero (4 of 18 cells). |
| `outputs/charts/conditional_sharpe_smoothed.png` | the same under smoothed labels. |
| `outputs/charts/timed_vs_static.png` | cumulative net return of static, timed-filtered and timed-smoothed at the headline η, lag and cost, on one log axis. |

`outputs/charts/features_review.png` (step 2.4) is the ten raw feature series
and is a data check rather than a result.

---

## The headline row, and the grid it sits in

```
 eta  lag  cost_bp       source  sharpe_static  sharpe_timed   diff  diff_p05  diff_p95  p_one_sided  mean_turnover  n_months
 0.5    1       20 hmm_filtered       0.206378      0.179378 -0.027 -0.227855  0.170963       0.5865       0.043992       257
```

The grid is 3 η × 2 lags × 3 costs × 4 label sources = 72 rows in
`outputs/tables/timing_results.csv`. Counted over all 72:

| count | value |
|---|---|
| cells with `diff` > 0 | 20 of 72 |
| cells with `p_one_sided` < 0.10 | 0 of 72 (smallest 0.178) |
| cells with `p_one_sided` < 0.05 | 0 of 72 |
| cells whose `[diff_p05, diff_p95]` excludes 0 | 7 of 72 |
| `diff` range | −0.381 to +0.073 |

All seven intervals that exclude zero are `rules`-baseline cells at η ≥ 0.25
and 20 to 50 bp, and all seven exclude zero **below**: the rules quadrant
traded hard enough loses significantly. Not one cell of the grid, under any
label source, is distinguishable from zero on the side of a gain.

---

## Robustness and fragility

Six variants completed, `outputs/tables/robustness/robustness_summary.csv`:

```
variant  sharpe_static  sharpe_timed      diff  diff_p05  diff_p95  p_one_sided  mean_turnover  fallback_share  n_months  n_pairwise_excl_zero  n_changes_on_refit_dates
 10feat       0.227102      0.143959 -0.083143 -0.304435  0.149639       0.7260       0.038276        0.393939       197                     5                         4
   diag       0.206378      0.198853 -0.007525 -0.185471  0.173890       0.5185       0.033943        0.279070       257                     2                         3
     k2       0.206378      0.110661 -0.095716 -0.298572  0.117894       0.7695       0.028156        0.186047       257                     2                         3
     k3       0.206378      0.179378 -0.027000 -0.227855  0.170963       0.5865       0.043992        0.279070       257                     1                         9
     k4       0.206378      0.103890 -0.102487 -0.295872  0.084231       0.8180       0.028621        0.337209       257                     4                        10
nolevel       0.206378      0.204965 -0.001413 -0.193397  0.173450       0.5125       0.039222        0.279070       257                     3                         7
```

No variant produces a positive `diff` and none reaches `p_one_sided` below
0.5125. The `k3` row reproduces the main headline row to every digit printed,
which is the check that the variant code path and the main code path are the
same code. The null does not depend on K, on the covariance structure, on the
feature set, or — from `outputs/tables/timing_results_blocksize.csv` — on the
bootstrap's block size, over which `p_one_sided` moves 0.011 across a factor
of four.

Fragility, `outputs/tables/robustness/fragility_summary.csv`:

```
          statistic  full_value   loo_min   loo_max  n_sign_flips  share_sign_flips  trimmed_value  n_months
      headline_diff   -0.027000 -0.049947  0.012667             1          0.003891       0.044294       257
umd_0_2_sharpe_diff   -1.024094 -1.156142 -0.822955             0          0.000000      -1.209043       197
```

**One month of 257 flips the headline's sign.** Removing 2026-07-31 — the last
month of the sample and the largest timed-minus-static month — takes `diff`
from −0.027 to +0.013, and trimming the 5% most influential months takes it to
+0.044. That trimmed value is a fragility diagnostic and is not a result: it
carries no interval, it is not a cell of any grid, and it is not what the
project found. It is reported because a point estimate one observation away
from changing sign is exactly the sort of number that should not be quoted
without it. The project found `diff = −0.027`, `p_one_sided = 0.5865`.

The one pairwise difference that excludes zero is not fragile the same way: it
flips on no single month, and the 5% trim moves it further from zero, to
−1.209.

---

## What did not work

- **The classifier moves when it is refitted, not when the data moves.** 9 of
  20 filtered label changes land on a refit date, 45%
  (`outputs/tables/classifier_diagnostics.csv`). Chaining state identity
  (convention 16) reduced this from 15 of 26 under a sort rule, so what remains
  is re-estimation rather than relabelling — but 45% against a 1-in-12 base
  rate is the central weakness of the whole exercise.
- **The anchor feature was too weak to order states on.**
  `outputs/tables/anchor_chain.csv` carries `sort_rule_state` beside the
  chained state: they disagree at **48 of 66** state-refit rows. The Hungarian
  chaining is what makes `param_drift.csv` readable as drift; sorting on
  `dgs10_chg12` was ordering noise.
- **`detects_2008` is false for both feature sets** (`classifier_diagnostics.csv`).
  This is a fault in the criterion, not in the classifier: the filter entered
  the low-rate, steep-slope, high-`vix` state at 2007-09-30 and held it through
  2008, so a K = 3 filter that was already in the right state had nowhere to
  move between 2008-06-30 and 2008-11-30. It is left unchanged rather than
  redefined after the fact, and the pre-registered rule that consumed it was
  not re-run.
- **One state is absorbing at the first refit.** `expected_duration.csv` has
  `inf` for 2004-12-31 state 0 — a self-transition probability of exactly 1 on
  168 training rows. Reported, not clipped.
- **K = 5 cannot complete the expanding protocol.**
  `outputs/tables/robustness/k_variant_outcomes.csv`:

  ```
   K  n_free_parameters              status                                                                                detail
   2                 90           completed                                                                                   NaN
   3                138           completed                                                                                   NaN
   4                188           completed                                                                                   NaN
   5                240 singular_covariance When `allow_singular is False`, the input matrix must be symmetric positive definite.
  ```

  240 free parameters trained on 192 rows collapses a state onto a
  7-dimensional hyperplane of the 8-dimensional feature space, and
  `forward_filter` refuses it. This was `decisions/OPEN.md` item 1 and was
  resolved by the reviewer as Option A: report the failure, change nothing.
  **The failure is the finding**; a filter that evaluated a density on a
  rank-deficient covariance would have hidden it on the main run's own code
  path.
- **The 10-feature variant's states are the least stable in the project.** 20
  of its 51 chained states matched at more than 1 z, the largest at 3.80
  (`outputs/tables/robustness/10feat/anchor_chain.csv`), and it is the only
  variant whose `sharpe_static` differs, because its window starts at its own
  2009-12-31 first refit (197 months against 257).
- **The non-primary feature set changes nothing.** Step 6.6's rerun is the
  `nolevel` row of `robustness_summary.csv` above: `diff` −0.001413,
  `p_one_sided` 0.5125, beside the main run's −0.027000 and 0.5865. Both are
  the same answer.
- **One decision date is dropped.** `model_input` drops 2026-02-28 because
  `cpi_m3` is NaN in the as-of-2026-02-28 vintage — October 2025 CPI was never
  published in that vintage. The row is reported in `dropped_rows.csv`, not
  patched.
- **Nothing failed to converge.** Across the 22 main refits, 440 of 440 HMM
  restarts and 440 of 440 GMM restarts converged
  (`outputs/tables/hmm_restarts_<refit>.csv`,
  `outputs/tables/gmm_restarts_<refit>.csv`), and no main-run state was
  degenerate (`n_degenerate_states` = 0 in `classifier_diagnostics.csv`). The
  null is not an optimisation failure.

The full diagnostics table, `outputs/tables/classifier_diagnostics.csv`:

```
  feature_set  n_filtered_changes  n_changes_on_refit_dates  share_on_refit_dates  median_run_months  max_expected_duration  n_infinite_durations  n_degenerate_states  max_matched_distance  detects_2008  detects_2020  filtered_smoothed_agreement
         core                  20                         9              0.450000               10.0             129.875370                     1                    0              2.689391         False          True                     0.682171
core_no_level                  21                         7              0.333333                9.5              57.102728                     0                    0              2.131451         False         False                     0.403101
```

The pre-registered rule that consumed it, quoted from
`decisions/primary_feature_set.md`, which quotes
`instructions/03b_section_3_revision.md`:

> Decision rule, applied mechanically: primary becomes "core_no_level" if and
> only if the d = 7 row has detects_2008 true AND detects_2020 true AND
> n_changes_on_refit_dates strictly below the d = 8 row's. Otherwise primary
> stays "core".

The d = 7 row fails the first two conditions, so the third — which it passes,
7 against 9 — never applies. Primary stays `"core"`, d = 8. The rule was fixed
before any factor return was loaded and was not rewritten once its outcome was
known.

---

## Limitations

- **Vintage coverage is not perfect.** Each of CPIAUCSL, INDPRO and UNRATE has
  its latest observation month at t−1 for **438 of 440 decision dates
  (99.55%)**; the exceptions are 1996-01-31 and 2025-11-30 for CPI and
  2025-10-31 and 2025-11-30 for INDPRO and UNRATE — all four are delayed
  releases around the 1995-96 and 2025 federal shutdowns
  (`review/section_1.md`, step 1.4 evidence, which prints the as-of rows these
  come from). No decision date has no vintage at all. The 2026-02-28 drop above is the one case where a missing
  release removed a row outright.
- **Two market series do not span the sample.**
  `outputs/tables/feature_sanity.csv` counts the NaN month-ends per feature:
  `breakeven_chg12` 168, `dgs10_chg12`, `dollar_chg12` and `oil_chg12` 12 each
  (their first 12-row change), and `dgs10_level`, `slope_2s10s`, `cpi_3m_ann`,
  `indpro_chg12`, `log_vix` and `unrate_chg12` none. `breakeven_chg12` is why
  the 10-feature variant starts from 2004-01-31 and not from 1991, and why it
  is the only variant on a different out-of-sample window.
- **Some regimes are thin.** Out-of-sample months per filtered state are
  **91 / 61 / 106** and per smoothed state **73 / 38 / 147** (the `n` column of
  `conditional_stats_hmm_filtered.csv` and
  `conditional_stats_hmm_smoothed.csv`). The refit split cuts these further:
  `conditional_refit_split_differences.csv` has cells resting on `n_true` = 12.
  A Sharpe on 12 months is a number, not a measurement, and the sample size is
  printed beside every conditional statistic for that reason.
- **The strategy spends a quarter of its life as its own comparator.** At
  `min_regime_obs` = 24, 72 of 258 months (27.9%) fall back to static weights
  (`outputs/tables/robustness/minobs_fallback.csv`). That is the price of not
  setting a weight from a handful of months in a barely-seen regime, not a
  defect to tune away — at 12 the fallback share halves to 14.0% and turnover
  rises, which is more trading rather than more signal.
- **Sleeve-internal turnover is not modelled** (convention 9). Costs cover only
  the turnover of the sleeve weights, c × ½ Σ|Δw|, which at the headline cell
  averages 0.043992 per month (`timing_results.csv`). Each Fama-French factor
  rebalances internally, and that cost is identical across the timed and the
  static book and is excluded from both. A live implementation would pay it,
  and it would not change the difference between them.
- **One sample, one country, one factor library.** The out-of-sample window is
  257 or 258 months of US data ending 2026-07-31. The UK and euro-area second
  panel the outline offered as optional robustness was not built.

---

## Methods note

**The as-of vintage rule.** Decision date t is the last calendar day of month
t. Market series take the last non-missing daily observation dated on or before
t, looking back at most 10 calendar days, else NaN. Revised series (CPIAUCSL,
INDPRO, UNRATE) take the ALFRED vintage as of t: for each observation month,
the release with the greatest `realtime_start` ≤ t. **All lags of a revised
series come from that same as-of-t vintage** — CPI_m and CPI_{m−3} are both
read from the vintage as of t, never from the latest vintage and never from a
mix (convention 12). A decision made at t is executed at the close of the first
trading day of t+1 and earns month t+1 (lag 0) or t+2 (lag 1). Lags are counted
in rows of the decision-date index, not in calendar days.

**Chained anchoring.** `anchor_permutation` orders the first refit
(2004-12-31) by the state mean of `dgs10_chg12` and nothing else. Every later
refit is relabelled by `chain_permutation`: `linear_sum_assignment` on the
Euclidean distance matrix between the previous refit's anchored state means and
the new state means, over all model-input columns, so new state j is the state
matched to previous anchored state j. The GMM chains from the HMM's first
anchored refit and the smoothed fit chains to the last expanding refit, so
every frame numbers the same regime the same way. The matched distance per
state per refit is in `anchor_chain.csv` beside `sort_rule_state`; a distance
above 1.0 z is reported and never acted on.

**The pre-registered feature-set rule.** The choice between the d = 8 and d = 7
feature sets was fixed as a mechanical rule on
`outputs/tables/classifier_diagnostics.csv` — quoted in full under "What did
not work" — after the first section 3 run showed that the d = 8 filter did not
change state through 2008, and **before any factor return was loaded**. It was
applied by `regime.tables.primary_feature_set_decision` and not rewritten once
its outcome was known. The set it rejected is rerun in step 6.6.

**The two step 6.7 definitions.** Both were fixed by implementation and
accepted by the reviewer (`decisions/section_6_review.md`, Q2).

1. *The trim count is `floor(0.05 n)`.* At n = 257 that removes 12 months
   (4.67%), not 13 (5.06%). A floor, so "the 5% of months" never removes more
   than 5% of them.
2. *Pairwise influence is measured by how far a month's removal moves the
   statistic* (`loo_diff − full_diff`). For the headline there is a timed and a
   static return in the same month, so the largest absolute timed-minus-static
   difference is a per-month quantity. For a pairwise state difference the two
   states occupy **disjoint** months and no such pairing exists, so the
   leave-one-out movement is what influence has to mean.

**Reproducibility.** `run.seed` is 20260917; restart i of any fit uses
`random_state = seed + i` and every bootstrap uses `seed`. Nothing uses default
randomness. Two consecutive `python -m regime.run` invocations on one machine
write byte-identical files under `outputs/` except
`outputs/tables/runtime.csv`, which is appended once per section per run by
design; the comparison table is in `review/section_7.md`. That is a
same-machine property — floating-point output moves at the 1e-12 level across
Python, BLAS and library versions — and is not a claim that a different
environment produces the same bytes.
