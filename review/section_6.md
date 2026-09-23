# Review — Section 6: Robustness

## Section

Section 6 of `PLAN.md` — steps 6.1 to 6.6 as planned, plus the new step 6.7
the reviewer added in answer to section 5 Q3. It asks whether the project's
result survives the choices made along the way: the number of states, the
covariance structure, the feature set, the length of the sample, the fallback
threshold, the bootstrap's block size, and finally the influence of individual
months. **Every step completed.** The section did not stop early and hit no
rule 4 threshold; `pytest -q` passed at the end of every step and passes now
at **116 passed**.

One variant inside step 6.1 could not be computed. `K = 5` fits but cannot be
filtered: the 2006-12-31 refit collapses a state onto 7 training rows in 8
dimensions and `forward_filter` refuses the rank-deficient Gaussian. That is
`decisions/OPEN.md` item 1, raised with two options and neither taken (rule
1). `K ∈ {2, 3, 4}` completed, and nothing else in the section depended on
K = 5.

**Nothing in this section is the project's result.** The project's result is
the headline cell of the main run, fixed in `config.toml` before section 5 ran
— η 0.5, lag 1, 20 bp, `hmm_filtered`, `diff = −0.027`, `p_one_sided =
0.5865`. Every number below is reported against that cell and none of them
replaces it.

## Steps completed

- `06: instructions` — `dc74643`
- `section 6: reviewer decisions and step 6.7` — `722ad1d`
- `step 6.1: K in 2, 3, 4, 5 rerun, and the one K that cannot be filtered` — `9c9adc2`
- `step 6.2: the diagonal-covariance rerun, and the covars_ round trip it needed` — `276cf8c`
- `step 6.3: the ten-feature run, from 2004-01-31 with a 2009-12-31 first refit` — `e899d30`
- `step 6.4: min_regime_obs over 12, 24, 36, with the fallback share beside the diff` — `4524c3c`
- `step 6.5: bootstrap block_size over 3, 6, 12, for the headline cell only` — `4f68590`
- `step 6.6: the non-primary feature set's conditional statistics, bootstrap and grid` — `86281f1`
- `step 6.7: how much of each headline statistic one month is carrying` — `a47a3c3`

## Evidence

### Every variant's headline-equivalent row

`outputs/tables/robustness/robustness_summary.csv` in full:

```
variant  sharpe_static  sharpe_timed      diff  diff_p05  diff_p95  p_one_sided  mean_turnover  fallback_share  n_months  n_pairwise_excl_zero  n_changes_on_refit_dates
 10feat       0.227102      0.143959 -0.083143 -0.304435  0.149639       0.7260       0.038276        0.393939       197                     5                         4
   diag       0.206378      0.198853 -0.007525 -0.185471  0.173890       0.5185       0.033943        0.279070       257                     2                         3
     k2       0.206378      0.110661 -0.095716 -0.298572  0.117894       0.7695       0.028156        0.186047       257                     2                         3
     k3       0.206378      0.179378 -0.027000 -0.227855  0.170963       0.5865       0.043992        0.279070       257                     1                         9
     k4       0.206378      0.103890 -0.102487 -0.295872  0.084231       0.8180       0.028621        0.337209       257                     4                        10
nolevel       0.206378      0.204965 -0.001413 -0.193397  0.173450       0.5125       0.039222        0.279070       257                     3                         7
```

Against the main run's headline row, which `timing_results.csv` carries:

```
 eta  lag  cost_bp       source  sharpe_static  sharpe_timed   diff  diff_p05  diff_p95  p_one_sided  mean_turnover  n_months
 0.5    1       20 hmm_filtered       0.206378      0.179378 -0.027 -0.227855  0.170963       0.5865       0.043992       257
```

### No variant clears `p_one_sided` 0.1, and none has a positive `diff`

The instruction message asked for a confirmation check on any variant whose
headline-equivalent `diff` came in below `p_one_sided` 0.1. Sorted ascending:

```
variant      diff  p_one_sided
nolevel -0.001413       0.5125
   diag -0.007525       0.5185
     k3 -0.027000       0.5865
 10feat -0.083143       0.7260
     k2 -0.095716       0.7695
     k4 -0.102487       0.8180
```

```
variants with p_one_sided < 0.1: 0
variants with diff > 0: 0
```

The lowest is 0.5125 and **every variant's point estimate is negative**. The
check was therefore not triggered. The three conditions it names are
nonetheless satisfied by construction and were verified; see "The three
conditions, verified anyway" below.

### `k3` reproduces the main run exactly

`K = 3` is `primary_K`, so step 6.1's `k3` variant runs the main run's model
through the variant code path. The headline row of
`outputs/tables/timing_results_k3.csv`:

```
 eta  lag  cost_bp       source  sharpe_static  sharpe_timed   diff  diff_p05  diff_p95  p_one_sided  mean_turnover  n_months
 0.5    1       20 hmm_filtered       0.206378      0.179378 -0.027 -0.227855  0.170963       0.5865       0.043992       257
```

Identical to the main run's headline row above, to every digit printed. That
is the check that the variant path and the main path are the same code and
not two implementations that happen to agree.

### Step 6.1 — K ∈ {2, 3, 4}, and why K = 5 is not among them

`outputs/tables/robustness/k_variant_outcomes.csv`:

```
 K  n_free_parameters              status                                                                                detail
 2                 90           completed
 3                138           completed
 4                188           completed
 5                240 singular_covariance When `allow_singular is False`, the input matrix must be symmetric positive definite.
```

The cause, from `outputs/tables/robustness/k5/state_counts.csv` — the only
degenerate row in 110 across all four K values:

```
refit_date  state  n_rows  degenerate
2006-12-31      4       7        True
```

Seven training rows in eight dimensions: the fitted covariance cannot have
rank above 7, and `scipy.stats.multivariate_normal` with
`allow_singular=False` refuses it. Measured directly at that refit, state 4's
covariance eigenvalues are

```
2.170e-16  2.506e-03  7.736e-03  8.326e-02  1.292e-01  7.726e-01  1.629e+00  3.379e+00
```

— a condition number of 1.6e16 on a matrix symmetric to 2.2e-16, so this is
rank deficiency and not a symmetry artefact. `hmm.min_covar` (1e-3) floors the
diagonal and cannot prevent it. The model has 240 free parameters and that
refit trains on 192 rows.

`select_k` completes at K = 5 because it fits only the 168-row first window
and never filters forward with those parameters, which is why section 3 did
not see this.

Per-K classifier behaviour across the four runs (22 refits each):

```
 K  refits  degenerate_rows  min_n_rows  anchor_disagreements  chain_rows  max_matched_distance
 2      22                0          67                     6          44              0.921301
 3      22                0          36                    48          66              2.689391
 4      22                0          26                    58          88              2.971731
 5      22                1           7                    82         110              2.678152
```

`anchor_disagreements` counts rows where the chained state number differs from
what the sort rule would have given that slot — 6 of 44 at K = 2 rising to 82
of 110 at K = 5, which is convention 16's premise (the sort was ordering
noise) getting worse as K rises.

### Step 6.2 — diagonal covariance

Headline-equivalent row, beside the main run's:

```
variant  sharpe_static  sharpe_timed      diff  diff_p05  diff_p95  p_one_sided  mean_turnover  fallback_share  n_months
   diag       0.206378      0.198853 -0.007525 -0.185471  0.173890       0.5185       0.033943        0.279070       257
   main       0.206378      0.179378 -0.027000 -0.227855  0.170963       0.5865       0.043992        0.279070       257
```

Parameter drift of the anchor feature `dgs10_chg12`, from each run's own
`param_drift.csv` — the largest spread of any one anchored state's mean across
the 22 refits, in z-units:

```
 run  min_mean  max_mean  max_abs_drift_across_refits  state_of_max_drift
main -0.596893  0.496341                     0.924760                   0
diag -0.592450  0.915583                     0.926943                   0
```

The drift is essentially the same, 0.925 against 0.927 z, and in both runs it
is state 0 that moves most. The full per-refit state means for both runs are
in `outputs/tables/param_drift.csv` and
`outputs/tables/robustness/diag/param_drift.csv`; the diagonal run's states
separate more sharply on this feature (its state 1 sits near −0.42 to −0.49 z
for most of the 2010s, where the main run's three states all sit inside ±0.5
z), which is what removing the off-diagonal terms buys and is not a timing
result.

### Step 6.3 — the ten-feature run

First-window restarts, `outputs/tables/robustness/10feat/hmm_restarts_2009-12-31.csv`
in full — 20 of 20 converged, the kept restart being restart 16 at
−692.428004:

```
 restart     seed      loglik  n_iter  converged
       0 20260917 -692.428006      33       True
       1 20260918 -706.849295       6       True
       2 20260919 -706.849302       5       True
       3 20260920 -706.849295       6       True
       4 20260921 -706.849333       6       True
       5 20260922 -695.479971      14       True
       6 20260923 -693.899525      17       True
       7 20260924 -706.849336       8       True
       8 20260925 -693.899525      12       True
       9 20260926 -694.159267      13       True
      10 20260927 -706.849294       7       True
      11 20260928 -706.849335       9       True
      12 20260929 -706.849336       8       True
      13 20260930 -706.849265       7       True
      14 20260931 -692.428007      16       True
      15 20260932 -706.849283       7       True
      16 20260933 -692.428004      11       True
      17 20260934 -693.899514      74       True
      18 20260935 -706.849336       7       True
      19 20260936 -706.849295       5       True
```

First-window state counts — none degenerate, the smallest state holding 13 of
72 rows:

```
refit_date  state  n_rows  degenerate
2009-12-31      0      13       False
2009-12-31      1      26       False
2009-12-31      2      33       False
```

Anchor chaining over its 17 refits:

```
 refits  rows  anchor_disagreements  max_matched_distance  n_over_1z
     17    51                    32              3.798758        20
```

**20 of 51 chained states matched at more than 1 z**, against a maximum of
3.80. That is reported, never acted on (convention 16). It is worse than any
run in step 6.1 and is the strongest single signal in this section that the
d = 10 variant's states are not stable objects across refits.

Its headline-equivalent row covers **197** earning months rather than 257,
because its out-of-sample window starts at its own first refit (2009-12-31):

```
variant  sharpe_static  sharpe_timed      diff  diff_p05  diff_p95  p_one_sided  mean_turnover  fallback_share  n_months
 10feat       0.227102      0.143959 -0.083143 -0.304435  0.149639        0.726       0.038276        0.393939       197
```

`sharpe_static` differs from every other variant's (0.2271 against 0.2064)
for the same reason: a static book over 197 months is not the same sample as
one over 257. This variant is comparable with the main run only as a
direction, not as a level.

### Step 6.4 — `min_regime_obs`, and the reviewer's Q2 reporting requirement

Headline row per N:

```
 min_regime_obs  sharpe_static  sharpe_timed      diff  diff_p05  diff_p95  p_one_sided  mean_turnover  n_months
             12       0.206378      0.184375 -0.022003 -0.227865  0.185686       0.5700       0.054712       257
             24       0.206378      0.179378 -0.027000 -0.227855  0.170963       0.5865       0.043992       257
             36       0.206378      0.150758 -0.055619 -0.233050  0.129391       0.7010       0.035424       257
```

`outputs/tables/robustness/minobs_fallback.csv` for the headline source and
η — the columns the reviewer required beside the `diff`:

```
 min_regime_obs       source  eta  n_months  n_fallback  fallback_share  mean_abs_deviation
             12 hmm_filtered  0.5       258          36        0.139535            0.091279
             24 hmm_filtered  0.5       258          72        0.279070            0.077755
             36 hmm_filtered  0.5       258         108        0.418605            0.060953
```

Read together, these say what the reviewer's answer said to look for. Going
from N = 24 to N = 12 improves the `diff` from −0.0270 to −0.0220, and over
the same move the fallback share halves (27.9% to 14.0%), the mean absolute
weight deviation rises (0.0778 to 0.0913) and the mean turnover rises (0.0440
to 0.0547). **The `diff` is least negative where the book trades most.** That
is more trading, not more signal; `p_one_sided` stays at 0.57. The fallback
share does not say otherwise.

`n_months` is 258 in the fallback table and 257 in the grid: the fallback
share is a property of the decision dates, of which there are 258, and the
grid counts earning months, of which the last decision date has none at lag 1.

N = 24 is the configured value, and its file is `timing_results.csv`:

```
                       file                                                           sha256
         timing_results.csv 23b9905aa925054dc8e3f7e18acc37cc8c7577661780cfde861ae57a8432189a
timing_results_minobs24.csv 23b9905aa925054dc8e3f7e18acc37cc8c7577661780cfde861ae57a8432189a
                  identical                                                             True
```

### Step 6.5 — block size

`outputs/tables/timing_results_blocksize.csv` in full:

```
 block_size  eta  lag  cost_bp       source  sharpe_static  sharpe_timed   diff  diff_p05  diff_p95  p_one_sided  mean_turnover  n_months
          3  0.5    1       20 hmm_filtered       0.206378      0.179378 -0.027 -0.215921  0.156870       0.5975       0.043992       257
          6  0.5    1       20 hmm_filtered       0.206378      0.179378 -0.027 -0.227855  0.170963       0.5865       0.043992       257
         12  0.5    1       20 hmm_filtered       0.206378      0.179378 -0.027 -0.224503  0.171390       0.5955       0.043992       257
```

`diff` is identical across the three rows by construction: only the resampling
varies, so the weight book and both net-return series are built once outside
the sweep. `p_one_sided` moves by 0.011 across a factor of four in block size.
**The interval's width is not what is keeping the headline `diff`
indistinguishable from zero.**

`outputs/tables/conditional_stats_blocksize.csv`, the same 4 of 18 cells
excluding zero at every block size:

```
 block_size  sum  count
          3    4     18
          6    4     18
         12    4     18
```

```
 block_size factor  state   n   sharpe  sharpe_p05  sharpe_p95
          3 Mkt-RF      1  61 1.087095    0.474167    1.761649
          3 Mkt-RF      2 106 0.712075    0.155356    1.346872
          3    RMW      0  91 0.944037    0.255610    1.657584
          3    UMD      0  91 0.761095    0.271447    1.266107
          6 Mkt-RF      1  61 1.087095    0.588614    1.773611
          6 Mkt-RF      2 106 0.712075    0.174042    1.282782
          6    RMW      0  91 0.944037    0.287502    1.646790
          6    UMD      0  91 0.761095    0.306703    1.246084
         12 Mkt-RF      1  61 1.087095    0.638233    1.774097
         12 Mkt-RF      2 106 0.712075    0.201883    1.211736
         12    RMW      0  91 0.944037    0.224347    1.763715
         12    UMD      0  91 0.761095    0.353849    1.256326
```

### Step 6.6 — the non-primary feature set

Headline row of `outputs/tables/timing_results_nolevel.csv`:

```
 eta  lag  cost_bp       source  sharpe_static  sharpe_timed      diff  diff_p05  diff_p95  p_one_sided  mean_turnover  n_months
 0.5    1       20 hmm_filtered       0.206378      0.204965 -0.001413 -0.193397   0.17345       0.5125       0.039222       257
```

against the main run's `diff = −0.027000`, `p_one_sided = 0.5865`. The d = 7
set comes closer to breaking even and is no more distinguishable from zero.

Conditional cells and pairwise differences excluding zero, both runs:

```
                         run  pairwise_rows  pairwise_excl_zero  cell_rows  cells_excl_zero
            main (core, d=8)             18                   1         18                4
nolevel (core_no_level, d=7)             18                   3         18                3
```

The d = 7 run's three:

```
factor  state_a  state_b  sharpe_diff  diff_p05  diff_p95  excludes_zero  n_a  n_b
   SMB        0        1     1.137444  0.171769  2.282811           True  105   37
   SMB        1        2    -1.168704 -2.393559 -0.097308           True   37  116
   UMD        0        2     0.940538  0.064410  1.626707           True  105  116
```

Three of 18 is what a 90% interval gives by chance (1.8 expected); one of 18,
the main run's count, is below it. Neither run separates the premia. **The
state numbers are not comparable between the two runs**: each feature set
anchors its own first refit and chains within itself (convention 16), so the
d = 7 UMD 0-against-2 difference of +0.94 and the d = 8 one of −1.02 are not
the same comparison with opposite signs — they are two different pairs of
states that happen to carry the same numbers.

### Step 6.7 — fragility

`outputs/tables/robustness/fragility_summary.csv` in full:

```
          statistic  full_value   loo_min   loo_max  n_sign_flips  share_sign_flips  trimmed_value  n_months
      headline_diff   -0.027000 -0.049947  0.012667             1          0.003891       0.044294       257
umd_0_2_sharpe_diff   -1.024094 -1.156142 -0.822955             0          0.000000      -1.209043       197
```

The 10 months whose removal moves the headline `diff` most, with their timed
and static net returns:

```
dropped_month      diff  sign_flipped      move  timed_net_ret  static_net_ret  timed_minus_static
   2026-07-31  0.012667          True  0.039667      -0.031394         0.00626           -0.037654
   2016-11-30 -0.001134         False  0.025866       0.002331         0.02816           -0.025829
   2008-06-30 -0.049947         False -0.022947       0.059757         0.03136            0.028397
   2023-12-31 -0.006959         False  0.020041      -0.009950         0.00970           -0.019650
   2008-07-31 -0.006996         False  0.020004      -0.012571         0.00702           -0.019591
   2021-02-28 -0.007172         False  0.019828      -0.015300         0.00410           -0.019400
   2020-01-31 -0.046466         False -0.019466       0.001852        -0.01590            0.017752
   2019-08-31 -0.045264         False -0.018264       0.017625        -0.00128            0.018905
   2023-11-30 -0.008920         False  0.018080      -0.019470        -0.00174           -0.017730
   2009-01-31 -0.044035         False -0.017036      -0.020062        -0.03242            0.012358
```

**Exactly one month of 257 flips the sign**: 2026-07-31, the last month of the
sample, which section 5 already named. It is also the largest
timed-minus-static month, at −3.77%. Dropping it moves `diff` from −0.0270 to
+0.0127.

The 5% trim (12 of 257 months) moves `diff` to **+0.0443**, further than any
single removal. It cuts the same way as the sign flip and not the other way:
a statistic that a dozen months out of 257 can move by 0.07 is small relative
to the influence of a handful of months, which is what a null looks like.
**The trimmed value is not a result and carries no interval.** The result
remains `diff = −0.027`, `p_one_sided = 0.5865`.

The same treatment for the one section 4 pairwise difference that excludes
zero — `hmm_filtered`, UMD, state 0 against state 2, found by reading
`conditional_differences_hmm_filtered.csv` rather than named in code:

```
dropped_month      diff  sign_flipped      move  state     ret
   2009-03-31 -0.822955         False  0.201138      2 -0.3436
   2018-12-31 -1.156142         False -0.132049      0 -0.0864
   2021-01-31 -1.143963         False -0.119870      0 -0.0799
   2007-12-31 -1.140137         False -0.116044      0 -0.0778
   2019-08-31 -1.121932         False -0.097838      0 -0.0674
   2008-05-31 -0.935612         False  0.088482      0  0.1273
   2020-10-31 -0.950954         False  0.073139      2 -0.1260
   2009-04-30 -0.951291         False  0.072803      2 -0.1254
   2008-11-30 -1.095233         False -0.071139      0 -0.0508
   2015-06-30 -1.095162         False -0.071068      2  0.1004
```

2009-03-31 is the month section 4's status file flagged, and it is indeed the
single most influential: a −34.36% UMD return in state 2, whose removal moves
the difference by +0.201. But **no single month flips the sign**, the whole
leave-one-out range [−1.156, −0.823] stays on one side of zero, and the 5%
trim (9 of 197 months) leaves it at −1.209, *more* negative than it started.
This one statistic is not carried by one month. That does not make it
evidence of regime separation — one of 18 pairwise differences excluding zero
is below the ~1.8 a 90% interval gives by chance, which is section 4's finding
and is unchanged.

### The three conditions, verified anyway

No variant triggered the sub-0.1 check, but the three conditions it names hold
for every variant by construction, and were checked:

- **Weights at t use only s < t.** Every variant calls the same
  `regime.strategy.weights`, which calls `trailing_conditional_sharpe`, which
  selects `joined.index < t` strictly.
  `tests/test_strategy.py::test_trailing_sharpe_is_strictly_before_t` plants a
  1e3 return in months t+1 and t+2 and asserts `S` and `n_k` do not move.
  Section 6 adds no second path to a weight.
- **The lag is applied as specified.** Every variant backtests through the
  same `regime.strategy.backtest`, whose shift is positional on an index
  `join_next_return` has already asserted to be a complete monthly month-end
  sequence. `test_backtest_three_month_literal` checks PLAN.md step 5.3's
  example at both lags.
- **The static comparator runs on identical earning months.** `timing_cells`
  asserts `timed.index.equals(static.index)` per cell and raises otherwise;
  `run_robustness_blocksize` and `headline_net_returns` each repeat that
  assertion for the single cell they build. Every variant's `sharpe_static` is
  0.206378 — one number — except `10feat`, whose window is genuinely shorter
  and whose `n_months` says so.

### Section 6 is idempotent

`section_6(cfg)` was run end to end after all seven steps were committed.
`git status --short` afterwards printed nothing: every file under `outputs/`
was rewritten byte-identically. Wall-clock 455.5 s.

## Tests run

```
$ .venv/Scripts/python.exe -m pytest -q
........................................................................ [ 62%]
............................................                             [100%]
116 passed in 17.17s
```

`pytest -q` passed at the end of every step: 104 after 6.1, 106 after 6.2, 108
after 6.3, 110 after 6.4, 111 after 6.5, 112 after 6.6, 116 after 6.7. No test
was deleted, skipped, marked xfail or loosened.

## Runtime per step

Machine: the Windows 11 laptop this repo is checked out on, `.venv` Python
3.11, single process, nothing else running.

| step | wall-clock | machine | notes |
|---|---|---|---|
| 6.1 | 5:39 | laptop | threshold 60 min. Includes the K = 5 attempt, which spent ~95 s before the filter refused it |
| 6.2 | 0:52 | laptop | threshold 60 min |
| 6.3 | 0:32 | laptop | threshold 60 min |
| 6.4 | 0:55 | laptop | threshold 20 min |
| 6.5 | 0:03 | laptop | threshold 20 min |
| 6.6 | 0:10 | laptop | threshold 60 min. Cheap because section 3 already fitted this feature set |
| 6.7 | 0:02 | laptop | threshold 20 min |
| section 6 end to end | 7:36 | laptop | |

**No step came close to its threshold.** Nothing was reduced:
`hmm.n_restarts` is the configured 20, `bootstrap.n_replications` the
configured 2000, and every cell of every grid was run.

**Step 6.7 was measured before being classified, as the instruction required.**
It takes 1.7 seconds, far under 20 minutes, so it is **not** added to
`run.long_steps` and `config.toml` is unchanged.

## Not verified

- **K = 5 is not covered by step 6.1.** Its classifier, conditional statistics
  and timing grid do not exist. `outputs/tables/robustness/k5/` holds the
  restart tables, `state_counts.csv` and `anchor_chain.csv` the refit loop
  produced before the filter refused the 2006-12-31 parameters; there is no
  `filtered_probs.csv` and no `timing_results_k5.csv`.
- **The step 6.1 and 6.6 variants' GMM and rules rows were not rerun**, as
  PLAN.md step 6.1 fixes: rules do not depend on K or on the covariance type,
  and the GMM is the comparison model at `primary_K` only. Each variant's
  timing grid is 36 rows over two sources, not 72 over four.
- **Step 6.2 does not recompute BIC**, per PLAN.md. Whether a diagonal model
  would have chosen a different K is not answered anywhere in this section.
- **The `10feat` variant is not comparable with the main run as a level.** Its
  197 earning months are a different sample; only the direction and the
  `p_one_sided` transfer.
- **Nothing in section 6 re-examines the main run's own numbers.** Sections 1
  to 5 were not re-run; their committed outputs are read as given, and the
  only code touched that they share is `model_from_params` (see below), whose
  `"full"` path returns its input unchanged.
- **The trimmed statistics carry no intervals.** `trimmed_diff` and
  `trimmed_pairwise_diff` are point values. Bootstrapping a trimmed statistic
  is not in `PLAN.md` and was not done.
- **`covars_for_setter` is exercised for `"full"` and `"diag"` only.** `"tied"`
  and `"spherical"` raise `NotImplementedError` by design; neither appears in
  `config.toml`.
- **Chart output.** Section 6 produces no charts; the robustness variants do
  not appear in `outputs/charts/`. Whether step 7.1's charts should carry them
  is a section 7 question.

## Open questions

One item was appended to `decisions/OPEN.md` this section.

**Item 1 — K = 5 cannot complete the expanding protocol (step 6.1).**

- **Option A** — report K = 5 as a failure of the variant and leave the code
  alone. Step 6.1 covers K ∈ {2, 3, 4}; `k_variant_outcomes.csv` carries K = 5
  with its status and parameter count; `hmm.k_grid` is unchanged so
  `bic_by_k.csv` still ranks all four. Cost: the robustness of the K choice is
  tested over a narrower range than planned.
- **Option B** — pass `allow_singular=True` in
  `regime/models/hmm_numpy.py::log_emissions`, so scipy evaluates the collapsed
  Gaussian through a pseudo-inverse and K = 5 completes. Cost: this changes the
  **main run's** filter, not a variant's. K = 3's covariances are well
  conditioned so its numbers would not move, but the filter would stop refusing
  a degenerate state anywhere, and a density evaluated on a rank-deficient
  covariance is not comparable across states.

**Steps left incomplete because of it:** none. Step 6.1 completed for
K ∈ {2, 3, 4} and every later step ran in full. Section 6 does not depend on
K = 5 anywhere.

## Files changed

**Before the steps**

- `instructions/06_section_6.md` — added (`dc74643`), the session's instruction
  message verbatim, rule 11.
- `decisions/section_5_review.md` — added (`722ad1d`), the reviewer's three
  answers.
- `PLAN.md` — modified (`722ad1d`): step 6.4 gains the fallback reporting
  requirement and its second test; step 6.7 added in full; step 7.4's first
  paragraph now states the null.

**Step 6.1** (`9c9adc2`)

- `regime/run.py` — `robustness_cfg` (generalising the old `_variant_cfg`),
  `robustness_dir`, `variant_classifier`, `variant_conditional`,
  `headline_row`, `fallback_summary`, `variant_summary_row`,
  `variant_timing_grid`, `append_robustness_summary`, `run_robustness_k`, and
  `section_6` replacing the not-built stub.
- `regime/tables.py` — `changes_on_refit_dates` extracted from
  `classifier_diagnostics_row`, which now calls it. One definition, two
  callers; the diagnostics row's numbers are unchanged.
- `tests/test_robustness.py` — added.
- `tests/test_run.py` — `test_unbuilt_section_raises` now asserts section 7 is
  the first unbuilt one. One line.
- `decisions/OPEN.md` — item 1 added.
- `outputs/tables/robustness/k{2,3,4,5}/`, `outputs/regimes/robustness/k{2,3,4}/`,
  `outputs/tables/timing_results_k{2,3,4}.csv`,
  `outputs/tables/robustness/k_variant_outcomes.csv`,
  `outputs/tables/robustness/robustness_summary.csv` — added.

**Step 6.2** (`276cf8c`)

- `regime/models/hmm.py` — `covars_for_setter` added; `model_from_params` calls
  it. `"full"` returns its input unchanged, so no main-run output moves.
- `regime/run.py` — `primary_k`, `run_robustness_diag`.
- `tests/test_robustness.py` — two tests appended.
- `outputs/tables/robustness/diag/`, `outputs/regimes/robustness/diag/`,
  `outputs/tables/timing_results_diag.csv` — added.

**Step 6.3** (`e899d30`)

- `regime/run.py` — `robustness_10feat_cfg`, `robustness_10feat_columns`,
  `run_robustness_10feat`.
- `tests/test_robustness.py` — two tests appended.
- `outputs/tables/robustness/10feat/`, `outputs/regimes/robustness/10feat/`,
  `outputs/tables/timing_results_10feat.csv` — added.

**Step 6.4** (`4524c3c`)

- `regime/run.py` — `MINOBS_FALLBACK_COLUMNS`, `run_robustness_minobs`.
- `tests/test_robustness.py` — two tests appended.
- `outputs/tables/timing_results_minobs{12,24,36}.csv`,
  `outputs/tables/robustness/minobs_fallback.csv` — added.

**Step 6.5** (`4f68590`)

- `regime/run.py` — `run_robustness_blocksize`.
- `tests/test_robustness.py` — one test appended.
- `outputs/tables/timing_results_blocksize.csv`,
  `outputs/tables/conditional_stats_blocksize.csv` — added.

**Step 6.6** (`86281f1`)

- `regime/run.py` — `nonprimary_feature_set`, `run_robustness_nonprimary`.
- `tests/test_robustness.py` — one test appended.
- `outputs/tables/robustness/nolevel/conditional_*.csv`,
  `outputs/tables/robustness/nolevel/unconditional_stats.csv`,
  `outputs/tables/timing_results_nolevel.csv` — added.

**Step 6.7** (`a47a3c3`)

- `regime/robustness.py` — added.
- `regime/run.py` — `headline_net_returns`, `excluding_zero_pairs`,
  `run_robustness_fragility`, `_pair_full_value`.
- `tests/test_robustness.py` — four tests appended.
- `outputs/tables/robustness/leave_one_month_out_headline.csv`,
  `leave_one_month_out_umd_0_2.csv`, `fragility_summary.csv` — added.

## Reviewer reads

1. **`outputs/tables/robustness/robustness_summary.csv`** — six rows, one per
   variant that completed. Look at `p_one_sided`: the lowest is 0.5125 and
   every `diff` is negative. Then look at `fallback_share` and
   `n_changes_on_refit_dates` beside each `diff`, which is what the columns
   are there for.
2. **`decisions/OPEN.md` item 1** — the one decision this section refused to
   take. Two options, and which steps it left incomplete (none).
3. **`outputs/tables/robustness/fragility_summary.csv`** and the two
   leave-one-out tables — two rows and their raw months. Check that the
   headline row's `trimmed_value` of +0.0443 is read as fragility and nowhere
   presented as a result.
4. **`outputs/tables/robustness/minobs_fallback.csv`** beside the three
   `timing_results_minobs<N>.csv` headline rows — the reviewer's own Q2
   requirement, and whether the "more trading, not more signal" reading is
   what the numbers support.
5. **`regime/run.py`, `section_6` and the seven `run_robustness_*` functions** —
   that every variant is a `dataclasses.replace` and no variant writes a main
   output path.
6. **`regime/models/hmm.py::covars_for_setter`** — the one change in this
   section to code the main run shares. Check that the `"full"` path returns
   its input unchanged.
7. **`regime/robustness.py`** — the two definitions fixed by implementation
   rather than by instruction: `_trim_count`'s floor, and the influence
   measure used for the pairwise trim. Both are documented at the point of
   definition and flagged in the status file.
