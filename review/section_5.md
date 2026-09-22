# Review — Section 5: the timed strategy

## Section

Section 5 of `PLAN.md`, steps 5.1 to 5.5: the trailing conditional Sharpe over
decision dates strictly before t, the weight vector and its three fallbacks,
the backtest indexed by the earning month, the 72-cell timing grid, and the
stationary-bootstrap interval and one-sided p on every cell of it. Every step
completed in one session. Nothing stopped early, nothing was refused under rule
1, nothing was appended to `decisions/OPEN.md`, and no step came near its
threshold — the whole section runs in about 25 seconds.

The session also carried a **section 4 addendum** first, answering the four
questions in `instructions/04_section_4.status.md` with the reviewer's answers
recorded in `decisions/section_4_review.md`: intervals on the refit split, and
`excess_sharpe` on every conditional table. Sections 1 to 4 were re-run with
no `--pull` and the regenerated tables committed with it. The section 4 numbers
already reported did not move.

**What the section found is a null, and that is the expected outcome.** The
headline cell — fixed in `config.toml` before any of this ran — is η 0.5, lag
1, 20 bp, `hmm_filtered`, and it gives `diff = −0.027`, interval
`[−0.228, 0.171]`, `p_one_sided = 0.5865`. Timing the five Fama-French sleeves
on the filtered HMM's regime labels did not beat holding them equally
weighted. That follows section 4, where the same classifier separated 1 of 18
pairwise state premia, below chance.

## Steps completed

- `05: instructions` — `15ea288`
- `section 4 addendum: refit-split intervals and excess Sharpe` — `4b0258d`
- `step 5.1: the trailing conditional Sharpe, over s < t strictly` — `a278dc9`
- `step 5.2: the weight vector, its three fallbacks and the static comparator` — `a2f56e4`
- `step 5.3: the backtest, indexed by the earning month` — `f355224`
- `step 5.4: the 72-cell timing grid, and what the weights actually do` — `a1279ca`
- `step 5.5: the timing-gain bootstrap, filling the interval and p columns` — `8d3ef4b`

## Evidence

### The out-of-sample window, and the one month missing from it

258 decision dates, 2004-12-31 to 2026-06-30. That span is 259 month-ends, so
one is absent:

```
expected months 259 actual 258 missing: [2026-02-28]
      date    missing
2026-02-28 cpi_3m_ann
```

2026-02-28 was dropped in section 2 for a missing `cpi_3m_ann`, so it carries
no label and cannot carry a weight. Its earning month is therefore absent from
every backtest: 2026-03-31 at lag 0, 2026-04-30 at lag 1.

```
lag 0: n=258, 2005-01-31 to 2026-07-31, missing [2026-03-31]
lag 1: n=257, 2005-02-28 to 2026-07-31, missing [2026-04-30]
```

The static and timed books are built on the identical decision dates and
backtested at the identical lag, so both carry the same gap and the comparison
stays like-for-like. `timing_cells` asserts the two indexes are equal for every
one of the 72 cells and raises otherwise. The lag-1 series is one row shorter
than the lag-0 one because the last decision date, 2026-06-30, has no month
t+2 in a factor file that ends 2026-07-31.

### Step 5.1 — `S` and `n_k` at 2010-12-31 and at the last decision date

`trailing_conditional_sharpe(hmm_filtered_labels, factors, t, cfg)`:

```
t = 2010-12-31  label_t = 2  assigned_t = True  n_k = 22
        trailing_sharpe
factor
SMB            0.942820
HML           -0.189457
RMW           -0.130756
CMA            0.767751
UMD           -1.110890

t = 2026-06-30  label_t = 1  assigned_t = True  n_k = 60
        trailing_sharpe
factor
SMB           -0.264854
HML            0.216312
RMW           -0.266869
CMA           -0.374865
UMD            0.321897
```

`sample.end` is 2026-07-31 and that decision date has no t+1 return, so the
last decision date the strategy ever acts on is 2026-06-30. The row before it:

```
t = 2026-05-31  label_t = 1  n_k = 59
        trailing_sharpe
factor
SMB           -0.357759
HML            0.152429
RMW           -0.155989
CMA           -0.466124
UMD            0.254994
```

`n_k` rises by exactly 1 from 2026-05-31 to 2026-06-30 because the two dates
share a label and exactly one new date entered the trailing sample. At
2010-12-31, `n_k = 22` is below `strategy.min_regime_obs` (24), so that date is
a fallback row however large `SMB`'s 0.943 looks.

### Step 5.2 — the fallback branches, per source and η

```
      source  eta  n_months  blended  unassigned  thin_regime  no_positive_sharpe  fallback_share
hmm_filtered 0.25       258      186           0           72                   0        0.279070
hmm_filtered 0.50       258      186           0           72                   0        0.279070
hmm_filtered 1.00       258      186           0           72                   0        0.279070
hmm_smoothed 0.25       258      186           0           72                   0        0.279070
hmm_smoothed 0.50       258      186           0           72                   0        0.279070
hmm_smoothed 1.00       258      186           0           72                   0        0.279070
gmm_filtered 0.25       258      191           0           67                   0        0.259690
gmm_filtered 0.50       258      191           0           67                   0        0.259690
gmm_filtered 1.00       258      191           0           67                   0        0.259690
       rules 0.25       258      154           0           96                   8        0.403101
       rules 0.50       258      154           0           96                   8        0.403101
       rules 1.00       258      154           0           96                   8        0.403101
```

The fallback share does not depend on η, which is correct: η scales the tilt
and cannot decide whether there is one to take.

`hmm_filtered` falls back on 72 of 258 months, every one of them through the
thin-regime branch, and the 72 split **exactly 24, 24, 24** across the three
states:

```
label
0    24
1    24
2    24
```

That is `strategy.min_regime_obs = 24` doing precisely what it was set to do:
the first 24 months the strategy has ever seen of each regime are held at 1/5.
The dates run from 2004-12-31 (the first out-of-sample date, `n_k = 0`) to
2023-03-31 — state 1 was still accumulating its 24th observation eighteen years
in. No date of any probabilistic source was unassigned, which matches section
4. `rules` is the only source that ever reaches the no-positive-Sharpe branch,
on 8 months.

### Step 5.2 — the branches reached, with `S` and `n_k` behind them

The first thin-regime date, which is also the first out-of-sample date:

```
branch 'thin_regime' at t = 2004-12-31: label_t = 0, assigned_t = True, n_k = 0, min_regime_obs = 24
     trailing_sharpe  weight
SMB              NaN     0.2
HML              NaN     0.2
RMW              NaN     0.2
CMA              NaN     0.2
UMD              NaN     0.2
```

`n_k = 0` at the first date is the strict `s < t` rule showing itself: there is
no history before the window opens, so the strategy has nothing to condition on
and holds 1/5.

The last blended date:

```
branch '' at t = 2026-06-30: label_t = 1, assigned_t = True, n_k = 60, min_regime_obs = 24
     trailing_sharpe    weight
SMB        -0.264854  0.100000
HML         0.216312  0.300955
RMW        -0.266869  0.100000
CMA        -0.374865  0.100000
UMD         0.321897  0.399045
```

Read against the formula at η = 0.5: `S+ = (0, 0.216312, 0, 0, 0.321897)`,
summing to 0.538209. The three negative Sharpes keep only `(1 − η)/5 = 0.1`
each; HML gets `0.1 + 0.5 × 0.216312 / 0.538209 = 0.300955` and UMD
`0.1 + 0.5 × 0.321897 / 0.538209 = 0.399045`. The row sums to 1.

For `hmm_filtered` at η 0.5 the unassigned and no-positive-Sharpe branches are
never reached, so those two are exercised on synthetic data by
`test_fallback_unassigned` and `test_fallback_no_positive_sharpe` rather than
on the real frames. That is recorded under "Not verified" below.

### Step 5.2 — how far the book ever actually moves from 1/5

A near-zero `diff` has two possible causes: the tilt was taken and did not pay,
or the tilt was never taken. `outputs/tables/weight_deviation.csv` separates
them.

```
      source  eta  n_months  n_months_off_static  share_off_static  mean_abs_deviation  max_abs_deviation
hmm_filtered 0.25       258                  186          0.720930            0.038877                0.2
hmm_filtered 0.50       258                  186          0.720930            0.077755                0.4
hmm_filtered 1.00       258                  186          0.720930            0.155510                0.8
hmm_smoothed 0.25       258                  186          0.720930            0.036380                0.2
hmm_smoothed 0.50       258                  186          0.720930            0.072760                0.4
hmm_smoothed 1.00       258                  186          0.720930            0.145520                0.8
gmm_filtered 0.25       258                  191          0.740310            0.040303                0.2
gmm_filtered 0.50       258                  191          0.740310            0.080606                0.4
gmm_filtered 1.00       258                  191          0.740310            0.161213                0.8
       rules 0.25       258                  154          0.596899            0.030334                0.2
       rules 0.50       258                  154          0.596899            0.060669                0.4
       rules 1.00       258                  154          0.596899            0.121337                0.8
```

The tilt was taken. `hmm_filtered` leaves 1/5 on 186 of 258 months (72.1%), and
at η 0.5 a weight sits on average 0.078 away from 0.2 with a maximum of 0.4 —
that maximum is a weight of 0.6, three times the static one, reached when a
single factor holds the whole positive-Sharpe mass. The maximum scales exactly
with η (0.2, 0.4, 0.8), which is the blend formula's ceiling `η × 4/5`.

So the null is not "the strategy never traded". It traded, and it did not pay.

### Step 5.3 — the static backtest at lag 1, 20 bp

```
            gross_ret  turnover  cost  net_ret
date
2005-02-28    0.01148       0.0   0.0  0.01148
2005-03-31    0.00580       0.0   0.0  0.00580
2005-04-30   -0.00872       0.0   0.0 -0.00872
...
2026-05-31   -0.02638       0.0   0.0 -0.02638
2026-06-30    0.02188       0.0   0.0  0.02188
2026-07-31    0.00626       0.0   0.0  0.00626
```

Maximum turnover over the whole static series: `0.0` exactly. A book that never
leaves 1/5 pays no cost, including on its first row, because the turnover
convention starts the book from the static vector rather than from cash.

### Step 5.4 — the headline cell and its static comparator

```
headline key (eta, lag, cost_bp, source): (0.5, 1, 20, 'hmm_filtered')
earning months identical: True | n = 257 | 2005-02-28 to 2026-07-31
```

Timed, first three and last three rows:

```
            gross_ret  turnover      cost   net_ret
date
2005-02-28   0.011480  0.000000  0.000000  0.011480
2005-03-31   0.005800  0.000000  0.000000  0.005800
2005-04-30  -0.008720  0.000000  0.000000 -0.008720
...
2026-05-31  -0.031382  0.006415  0.000013 -0.031395
2026-06-30   0.027237  0.152985  0.000306  0.026931
2026-07-31  -0.031257  0.068148  0.000136 -0.031394
```

The first rows are identical to the static book's because the first months are
thin-regime fallbacks.

The headline row of `timing_results.csv`, with the three columns step 5.5
filled:

```
 eta  lag  cost_bp       source  sharpe_static  sharpe_timed   diff  diff_p05  diff_p95  p_one_sided  mean_turnover  n_months
 0.5    1       20 hmm_filtered       0.206378      0.179378 -0.027 -0.227855  0.170963       0.5865       0.043992       257
```

`sharpe_static` is one number per (lag, cost_bp), identical across all four
sources and all three η, as `test_static_row_is_identical_across_sources`
requires:

```
             nunique     first
lag cost_bp
0   10             1  0.211710
    20             1  0.211710
    50             1  0.211710
1   10             1  0.206378
    20             1  0.206378
    50             1  0.206378
```

### Step 5.4 — cumulative net return of the headline cell

```
static: cumulative product of (1 + net_ret) at 2026-07-31 = 1.190447; sum of net_ret = 0.195200; sharpe = 0.206378
timed:  cumulative product of (1 + net_ret) at 2026-07-31 = 1.171871; sum of net_ret = 0.182784; sharpe = 0.179378
```

Last five months of both cumulative series:

```
            static_cum  timed_cum
date
2026-02-28    1.181118   1.205153
2026-03-31    1.189079   1.216310
2026-05-31    1.157711   1.178125
2026-06-30    1.183041   1.209853
2026-07-31    1.190447   1.171871
```

(2026-04-30 is the month the dropped 2026-02-28 decision date would have
earned; see the first subsection.)

The two books differ less in level than in spread:

```
static: mean*12 = 0.009114, std(ddof=1)*sqrt12 = 0.044164, sharpe = 0.206378, total cost = 0.000000
timed:  mean*12 = 0.008535, std(ddof=1)*sqrt12 = 0.047579, sharpe = 0.179378, total cost = 0.022612
```

The timed book earned 0.0006 a year less and carried 0.0034 more annualised
volatility, and paid 2.26 percentage points of cumulative trading cost over the
21 years. Concentrating into two of five sleeves raises the variance; the
regime signal did not raise the mean enough to pay for that, let alone for the
turnover.

### Step 5.4 — the ten largest single-month turnovers in the headline cell

```
      date  turnover
2015-02-28  0.447183
2021-04-30  0.446396
2020-06-30  0.440400
2017-02-28  0.428013
2014-02-28  0.417882
2020-02-29  0.417850
2021-02-28  0.413285
2016-12-31  0.408822
2020-09-30  0.397609
2016-11-30  0.377008
```

Against a mean of 0.044, the largest month is ten times the average. Four of
the ten fall in February and one on 31 December — consistent with section 3's
finding that the filtered labels move disproportionately at the December refit
(`decisions/section_3_review.md` Q1), whose effect on the weights lands one or
two months later.

### The headline result rests visibly on its last month

The five months where the timed book diverges most from the static one:

```
             static     timed  timed_minus_static  turnover
date
2026-07-31  0.00626 -0.031394           -0.037654  0.068148
2008-06-30  0.03136  0.059757            0.028397  0.016906
2016-11-30  0.02816  0.002331           -0.025829  0.377008
2023-12-31  0.00970 -0.009950           -0.019650  0.000000
2008-07-31  0.00702 -0.012571           -0.019591  0.002512
```

Removing that one month — the very last in the sample — reverses the sign of
the headline:

```
month removed: 2026-07-31
sharpe_static without it: 0.200220
sharpe_timed  without it: 0.212888
```

`diff` would be **+0.013** instead of −0.027. This is reported, not smoothed
over, and it cuts against the headline as much as for it: a gain of −0.027
that a single month out of 257 can flip is a null, not a negative finding. The
bootstrap says the same thing more carefully — `[−0.228, 0.171]` with
`p_one_sided = 0.5865` is as close to "no information" as this test can return.

### The 72-row grid

`outputs/tables/timing_results.csv`, committed in full. Summary counts over it:

```
rows: 72
p_one_sided < 0.1: 0
diff > 0: 20
interval excludes zero: 7
```

**Not one cell of the 72 has `p_one_sided` below 0.1.** 20 of 72 have a
positive point estimate and none of those 20 is distinguishable from zero. The
7 cells whose interval excludes zero are all `rules` and all **negative**:

```
 eta  lag  cost_bp source  sharpe_static  sharpe_timed      diff  diff_p05  diff_p95  p_one_sided  mean_turnover  n_months
0.25    0       50  rules       0.211710      0.102239 -0.109470 -0.225360 -0.011807       0.9685       0.035003       258
0.25    1       50  rules       0.206378      0.108538 -0.097839 -0.179919 -0.025885       0.9860       0.034985       257
0.50    0       50  rules       0.211710     -0.006738 -0.218448 -0.429693 -0.025840       0.9720       0.070006       258
0.50    1       20  rules       0.206378      0.065722 -0.140656 -0.296366 -0.001504       0.9505       0.069970       257
0.50    1       50  rules       0.206378      0.014605 -0.191773 -0.351477 -0.050031       0.9870       0.069970       257
1.00    0       50  rules       0.211710     -0.169229 -0.380939 -0.707398 -0.048709       0.9675       0.140012       258
1.00    1       50  rules       0.206378     -0.119665 -0.326042 -0.587318 -0.080924       0.9825       0.139941       257
```

The only statistically distinguishable effect in the entire grid is that the
backward-looking rules quadrants **destroy** Sharpe, increasingly so as η and
the cost rise. That is a coherent result rather than a curiosity: the rules
labels are the noisiest source, so they trade the most (mean turnover 0.140 at
η 1.0 against 0.088 for `hmm_filtered`) for the least signal.

`hmm_smoothed` — the hindsight labelling — has the largest positive point
estimates in the grid, up to +0.073 at η 0.5, lag 0, 10 bp, with
`p_one_sided = 0.249`. Even with the answers known in advance, perfect regime
knowledge does not produce a timing gain this test can distinguish from zero.
That is the strongest single statement the section makes, and it is the reason
none of the 20 positive cells should be read as a result.

### Step 5.5 — separating the cost of the lag from the cost of trading

`outputs/tables/timing_headline_decomposition.csv`. Source and η held at the
headline's; only the lag and the cost vary. The grid itself cannot show this,
because `strategy.cost_bp_grid` starts at 10 bp.

```
                label  eta  lag  cost_bp       source  sharpe_static  sharpe_timed      diff
             headline  0.5    1       20 hmm_filtered       0.206378      0.179378 -0.027000
      no lag, no cost  0.5    0        0 hmm_filtered       0.211710      0.230209  0.018499
no lag, headline cost  0.5    0       20 hmm_filtered       0.211710      0.208130 -0.003580
headline lag, no cost  0.5    1        0 hmm_filtered       0.206378      0.201703 -0.004675
```

The gross signal is worth `+0.0185` of Sharpe. 20 bp of trading costs takes
`0.0185 − (−0.0036) = 0.0221` of it; the one-month implementation lag takes
`0.0185 − (−0.0047) = 0.0232`. Together they take 0.0455 and leave −0.027. So
**the two costs are individually larger than the gross gain, and of almost
identical size.** Neither alone is the explanation; the signal is simply too
small to survive either.

### Step 5.5 — the timing gain is reproducible and signed correctly

`test_timing_gain_identical_series` gives `diff == 0`, `p05 == p95 == 0` and
`p_one_sided == 1.0` on a series bootstrapped against itself — exactly 0 in
every replication, not merely centred on 0, because both series travel in the
same bootstrap row. `test_timing_gain_reproducible` gives identical dicts on
two calls with `seed = cfg.run_seed`.
`test_timing_gain_is_signed_timed_minus_static` adds a constant 0.006 to the
timed series and gets a positive `diff` with `p_one_sided < 0.1`, so a real
gain would be detected if there were one.

`fill_timing_gain` re-derives each cell's point estimate inside the bootstrap
and raises if it disagrees with the `diff` `run_timing_grid` wrote; all 72 rows
agreed to 1e-12.

### Section 4 addendum — the refit split with intervals

`outputs/tables/conditional_refit_split_differences.csv`, first six of 18 rows:

```
factor  state  sharpe_diff  diff_p05  diff_p95  excludes_zero  n_false  n_true
Mkt-RF      0     1.506049 -1.898295  2.934918          False       22      69
Mkt-RF      1     0.450110 -1.310636  6.279401          False       49      12
Mkt-RF      2     0.115266 -2.099393  1.838937          False       23      83
   SMB      0    -0.399514 -1.541547  1.159996          False       22      69
   SMB      1     0.588653 -1.322367  2.727095          False       49      12
   SMB      2    -0.263690 -2.315605  1.482480          False       23      83
```

**0 of 18 refit-split differences exclude zero.** The large splits that
prompted the reviewer's Q2 — RMW state 0 at 3.228 against 0.263 — are noise on
these sample sizes. The per-half table `conditional_stats_refit_split.csv` has
10 of 36 halves whose own Sharpe excludes zero, but no pair of halves differs.
This is the same shape of answer as section 4's pairwise state differences: the
levels are sometimes distinguishable from zero, the differences are not.

### Section 4 addendum — `excess_sharpe`

`outputs/tables/conditional_stats_hmm_filtered.csv`, first six rows:

```
factor  state   n  ann_mean  ann_std    sharpe  excess_sharpe  sharpe_p05  sharpe_p95  excludes_zero
Mkt-RF      0  91  0.048303 0.140195  0.344543      -0.323032   -0.406935    1.344566          False
Mkt-RF      1  61  0.146656 0.134906  1.087095       0.419520    0.588614    1.773611           True
Mkt-RF      2 106  0.122672 0.172274  0.712075       0.044500    0.174042    1.282782           True
   SMB      0  91 -0.012976 0.086864 -0.149381      -0.115147   -0.712789    0.421516          False
   SMB      1  61 -0.027639 0.104311 -0.264971      -0.230737   -0.766503    0.364228          False
   SMB      2 106  0.019381 0.089124  0.217462       0.251697   -0.347618    0.787990          False
```

This is exactly the effect the reviewer asked to expose. Mkt-RF state 2 reads
0.712 and excludes zero, but the pooled Mkt-RF Sharpe over the same 258 dates
is 0.6676, so the excess is **+0.0445** — the regime added essentially nothing
to a factor that simply paid over this window. The same table's Mkt-RF state 1
is a genuine +0.420 of excess.

## Tests run

```
$ pytest -q
........................................................................ [ 71%]
.............................                                            [100%]
101 passed in 11.83s
```

Run at the end of every step, and at the end of the section. 101 passed, 0
failed, 0 skipped, 0 xfail. Section 5 added 18 tests in
`tests/test_strategy.py` and the addendum added 2 to `tests/test_conditional.py`.

## Runtime per step

| step | wall-clock | machine | notes |
|---|---|---|---|
| addendum | 0:24 (sections 1–4 re-run, no `--pull`) | Windows 11, local | section 4 alone is ~4 s |
| 5.1 + 5.2 | 0:02 | Windows 11, local | all four sources at all three η |
| 5.3 + 5.4 | 0:03 | Windows 11, local | 72 cells, weight books reused across the lag and cost axes |
| 5.5 | 0:10 | Windows 11, local | 72 cells × 2000 replications |
| section 5 end to end | 0:24 | Windows 11, local | `python -m regime.run --section 5` |

Every step is far under its threshold — 20 minutes, or 60 minutes for step 5.5.
Nothing was reduced: `bootstrap.n_replications` is the configured 2000,
`bootstrap.block_size` the configured 6, and all 72 cells were run.

## Not verified

- **The unassigned and no-positive-Sharpe fallback branches never fire on the
  real frames** for the probabilistic sources. No date of `hmm_filtered`,
  `hmm_smoothed` or `gmm_filtered` is unassigned (section 4 found the same),
  and no regime of theirs ever has every factor's trailing Sharpe at or below
  zero. Both branches are exercised only on synthetic data, by
  `test_fallback_unassigned` and `test_fallback_no_positive_sharpe`. `rules`
  reaches the no-positive-Sharpe branch on 8 months, so that branch does run on
  real data for one source.
- **No chart.** Section 5 writes tables only; the charts are step 7.1.
- **The cumulative-return paths are reported at their end values and last five
  months only.** No drawdown, no per-year decomposition — neither is in the
  plan for this section.
- **The internal turnover of each Fama-French sleeve is not modelled**
  (convention 9). The 2.26 percentage points of cumulative cost above is sleeve
  turnover only.
- **`conditional_stats_project1.csv` was not produced.** The project 1 file is
  absent, so the `excess_sharpe` path through `project1_conditional` is covered
  by `tests/test_project1.py::test_conditional_join_runs_when_present` on
  synthetic data and has never run on real project 1 returns.
- **The step 6.4 `min_regime_obs` sweep is not in this section.** The 24-month
  threshold holds 72 of 258 months at 1/5, which is a large share; whether the
  null is robust to 12 or 36 is step 6.4's question, not this one's.
- **No cell of the grid was checked against an independent implementation.**
  The backtest is verified against PLAN.md's literal three-month example and
  the timed-equals-static identity, not against a second code path.

## Open questions

None. Nothing was appended to `decisions/OPEN.md` this section.

## Files changed

**`05: instructions`** — added `instructions/05_section_5.md`.

**`section 4 addendum`** — added `decisions/section_4_review.md`,
`outputs/tables/conditional_refit_split_differences.csv`; modified
`regime/conditional.py` (`add_excess_sharpe`, `bootstrap_refit_split`,
`_sharpe_by_state_and_flag`), `regime/run.py` (section 4 order and the project
1 branch), `PLAN.md` (steps 4.2, 4.3, 7.4), `tests/test_conditional.py`,
`tests/test_project1.py`, and every regenerated table under `outputs/tables/`.

**`step 5.1`** — `regime/strategy.py` (+85), `tests/test_strategy.py` (+124).

**`step 5.2`** — `regime/strategy.py` (+78), `tests/test_strategy.py` (+79).

**`step 5.3`** — `regime/strategy.py` (+56), `tests/test_strategy.py` (+83).

**`step 5.4`** — `regime/strategy.py` (+106), `regime/run.py` (+98, section 5
and `headline_decomposition`), `tests/test_strategy.py` (+86),
`tests/test_run.py` (the unbuilt-section assertion moves from 5 to 6), and four
new tables: `timing_results.csv`, `weight_deviation.csv`,
`timing_headline_decomposition.csv`, `headline_turnover_top10.csv`.

**`step 5.5`** — `regime/strategy.py` (+80), `regime/run.py` (+2),
`tests/test_strategy.py` (+82), `outputs/tables/timing_results.csv` rewritten
with the interval and p columns filled.

## Reviewer reads

1. `outputs/tables/timing_results.csv` — the whole grid. Check that no cell has
   `p_one_sided` below 0.1 and that the only cells excluding zero are `rules`
   and negative.
2. `outputs/tables/timing_headline_decomposition.csv` — four rows. The gross
   signal is +0.0185 and each of the two costs is larger than it.
3. `outputs/tables/weight_deviation.csv` — that the tilt was actually taken:
   186 of 258 months off 1/5 for `hmm_filtered`, max deviation 0.4 at η 0.5.
4. `regime/strategy.py`, `trailing_conditional_sharpe` — the one place a
   lookahead could hide. The sample is `joined.index < t`, strictly.
5. `tests/test_strategy.py::test_trailing_sharpe_is_strictly_before_t` — the
   1e3 plant in months t, t+1 and t+2.
6. `tests/test_strategy.py::test_backtest_three_month_literal` — PLAN.md step
   5.3's example at both lags, every number written out.
7. `decisions/section_4_review.md` and
   `outputs/tables/conditional_refit_split_differences.csv` — the addendum:
   0 of 18 refit-split differences exclude zero.
8. The "headline result rests visibly on its last month" subsection above —
   the one place this section's number is fragile, and it is fragile in the
   direction that would flatter the strategy.
