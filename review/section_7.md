# Review — Section 7: Outputs and write-up

## Section

This is Section 7 of `PLAN.md` (steps 7.1 to 7.4): the three section 7 charts,
`regime_labels.csv` and the section 8 completeness check, the end-to-end
idempotence run from an empty `data/processed/`, and the README that answers
the four questions. **Every step completed**, in one session on 23 September
2026, without pausing between steps. No step hit rule 4, no step came near its
runtime threshold, and no new `decisions/OPEN.md` item was raised. Two commits
preceded the steps, as the instruction message required: the instruction saved
verbatim, and the reviewer's four section 6 answers recorded in
`decisions/section_6_review.md` with `OPEN.md` item 1 marked resolved under
Option A and kept rather than deleted.

The project's result is unchanged and nothing in this section could change it:
section 7 writes charts, tables and prose from outputs sections 1 to 6 already
produced. The headline cell is where `config.toml` fixed it before section 5
ran — η 0.5, lag 1, 20 bp, `hmm_filtered`, `diff` −0.027000,
`p_one_sided` 0.5865.

## Steps completed

- `07: instructions` — `c4f05b3`
- `section 7: reviewer decisions` — `c7f99ae`
- `step 7.1: the three charts, and the two heatmaps question 4 needs side by side` — `d9f3c26`
- `step 7.2: the published regime series, and a completeness check over the section 8 tables` — `508f8c1`
- `step 7.3: two end-to-end runs from an empty data/processed, compared file by file` — `e55b228`
- `step 7.4: the README, stating the null first and naming a committed table for every number` — `9c21f1d`

## Evidence

### The four PNGs exist and are non-empty

```
$ ls -la outputs/charts/
-rw-r--r-- 1 astha 197613       0 Sep 17 11:26 .gitkeep
-rw-r--r-- 1 astha 197613  151357 Sep 23 11:49 conditional_sharpe_filtered.png
-rw-r--r-- 1 astha 197613  145228 Sep 23 11:49 conditional_sharpe_smoothed.png
-rw-r--r-- 1 astha 197613 1038192 Sep 22 18:05 features_review.png
-rw-r--r-- 1 astha 197613  729913 Sep 23 11:49 regimes_timeline.png
-rw-r--r-- 1 astha 197613  364099 Sep 23 11:49 timed_vs_static.png
```

`features_review.png` is the step 2.4 chart and is rewritten by section 2;
the other four are step 7.1's. All five are written at
`cfg.outputs_chart_dpi` = 300 with `metadata={"Software": None}`.

### The `regimes_timeline.png` legend strings

Printed by `write_charts` from the last refit's `param_drift.csv` rows
(`state_legend_labels`, which sorts the refit dates and takes `iloc[-1]`):

```
$ python -c "... write_charts(load_config())"
INFO regime: regimes_timeline.png legend:
state 0: mean dgs10_chg12 = -0.03, mean cpi_3m_ann = -0.05
state 1: mean dgs10_chg12 = 0.32, mean cpi_3m_ann = 0.22
state 2: mean dgs10_chg12 = 0.06, mean cpi_3m_ann = -0.28
```

The rows they are formatted from, `outputs/tables/param_drift.csv` filtered to
the last refit date and the two anchor features:

```
refit_date  state     feature      mean  variance
2025-12-31      0 dgs10_chg12 -0.029357  0.674795
2025-12-31      0  cpi_3m_ann -0.051091  3.095571
2025-12-31      1 dgs10_chg12  0.318713  1.112450
2025-12-31      1  cpi_3m_ann  0.219358  1.008949
2025-12-31      2 dgs10_chg12  0.063702  0.751498
2025-12-31      2  cpi_3m_ann -0.282399  0.820981
```

The anchor feature spans −0.029 to 0.319 z across the three states at the last
refit — the span that convention 16 records as too narrow to sort on, visible
here in the legend the chart actually prints.

### The heatmap's borders equal the `excludes_zero` count

`conditional_sharpe_heatmap` returns its `Axes`; the only patches it adds are
the unfilled border rectangles.
`tests/test_charts.py::test_heatmap_border_only_where_excludes_zero` asserts
`len(borders) == len(ax.patches) == stats["excludes_zero"].sum()` and that the
bordered `(factor, state)` set equals the input's, over a synthetic table with
4 of 18 cells flagged, and asserts `ax.patches == []` when none is flagged.

On the real filtered table, 4 of 18 cells carry a border. The rows
(`outputs/tables/conditional_stats_hmm_filtered.csv`, `excludes_zero` rows):

```
factor  state   n  ann_mean  ann_std    sharpe  excess_sharpe  sharpe_p05  sharpe_p95  excludes_zero
Mkt-RF      1  61  0.146656 0.134906  1.087095       0.419520    0.588614    1.773611           True
Mkt-RF      2 106  0.122672 0.172274  0.712075       0.044500    0.174042    1.282782           True
   RMW      0  91  0.048541 0.051418  0.944037       0.496902    0.287502    1.646790           True
   UMD      0  91  0.094378 0.124003  0.761095       0.645527    0.306703    1.246084           True
```

### `regime_labels.csv` — head, tail and value counts per column

```
$ python -c "labels, checks = write_published_tables(load_config()); ..."
            rules_label  hmm_filtered_label  hmm_filtered_assigned  hmm_smoothed_label  gmm_filtered_label
date
1990-01-31         <NA>                <NA>                  False                <NA>                <NA>
1990-02-28         <NA>                <NA>                  False                <NA>                <NA>

            rules_label  hmm_filtered_label  hmm_filtered_assigned  hmm_smoothed_label  gmm_filtered_label
date
2026-06-30            2                   1                   True                   0                   2
2026-07-31            2                   1                   True                   0                   1
```

Value counts, including NaN, per column:

```
rules_label
0       116
1        97
2       122
3        91
<NA>     13

hmm_filtered_label
0        91
1        62
2       106
<NA>    180

hmm_filtered_assigned
False    180
True     259

hmm_smoothed_label
0       142
1        97
2       187
<NA>     13

gmm_filtered_label
0        95
1       145
2        19
<NA>    180
```

Reading them: the index is all 439 decision dates from `sample.start`
(1990-01-31) to `sample.end` (2026-07-31). The rules and smoothed sources have
labels from `features_from` onward, so 13 rows are empty — 1990-01-31 to
1990-12-31 is 12 rows, plus the single dropped date 2026-02-28. The two
filtered sources begin at the first refit 2004-12-31, so 180 rows are empty
(439 − 259). `hmm_filtered_assigned` is True on exactly the 259 dates with a
filtered label and False on all 180 without, which is the coverage rule the
step 7.2 test asserts. That every filtered date is `assigned` reflects how
sharply peaked the filtered probabilities are — `filtered_probs.csv` row 1 is
`1.0, 6.8e-251, 0.0` — not a threshold that never binds.

The written file, first and last rows and one row inside the rules-only span:

```
date,rules_label,hmm_filtered_label,hmm_filtered_assigned,hmm_smoothed_label,gmm_filtered_label
1990-01-31,,,False,,
2004-02-29,0,,False,2,
2026-07-31,2,1,True,0,1
```

Labels are written as nullable `Int64`, so state 0 reads `0` and not `0.0`; an
unavailable label is an empty field.

### `check_outputs` in full

`outputs/tables/check_outputs.csv`, 26 rows, every one present with the
specified columns:

```
                                           path  present  columns_ok
           outputs/tables/expected_duration.csv     True        True
                    outputs/tables/bic_by_k.csv     True        True
              outputs/tables/timing_results.csv     True        True
outputs/tables/transition_matrix_2004-12-31.csv     True        True
outputs/tables/transition_matrix_2005-12-31.csv     True        True
outputs/tables/transition_matrix_2006-12-31.csv     True        True
outputs/tables/transition_matrix_2007-12-31.csv     True        True
outputs/tables/transition_matrix_2008-12-31.csv     True        True
outputs/tables/transition_matrix_2009-12-31.csv     True        True
outputs/tables/transition_matrix_2010-12-31.csv     True        True
outputs/tables/transition_matrix_2011-12-31.csv     True        True
outputs/tables/transition_matrix_2012-12-31.csv     True        True
outputs/tables/transition_matrix_2013-12-31.csv     True        True
outputs/tables/transition_matrix_2014-12-31.csv     True        True
outputs/tables/transition_matrix_2015-12-31.csv     True        True
outputs/tables/transition_matrix_2016-12-31.csv     True        True
outputs/tables/transition_matrix_2017-12-31.csv     True        True
outputs/tables/transition_matrix_2018-12-31.csv     True        True
outputs/tables/transition_matrix_2019-12-31.csv     True        True
outputs/tables/transition_matrix_2020-12-31.csv     True        True
outputs/tables/transition_matrix_2021-12-31.csv     True        True
outputs/tables/transition_matrix_2022-12-31.csv     True        True
outputs/tables/transition_matrix_2023-12-31.csv     True        True
outputs/tables/transition_matrix_2024-12-31.csv     True        True
outputs/tables/transition_matrix_2025-12-31.csv     True        True
              outputs/regimes/regime_labels.csv     True        True
```

**What this does and does not cover.** The kickoff's section 8 names two
tables — "transition matrix, expected durations, BIC by K" and "timed minus
static Sharpe for each η, with and without lag, with bootstrap p-values" — and
its section 9 asks for the regime series as a published CSV. Those are the 26
rows above: 22 transition matrices (one per refit), `expected_duration.csv`,
`bic_by_k.csv`, `timing_results.csv` and `regime_labels.csv`.
`SECTION_8_TABLES` in `regime/tables.py` carries the exact column list for
each of the three named tables; the transition matrices are checked by glob,
for a `from_state` first column and `to_k` columns thereafter, because their
count is the number of refits and their width is K.

It does **not** cover the conditional, robustness or fragility tables, which
section 8 does not name. Those are written and asserted by sections 4 to 6 and
by the idempotence comparison below, which covers all 503 files under
`outputs/`.

### Step 7.3 — `runtime.csv` in full

`data/processed/` was emptied to `.gitkeep` before run 1 (the eight regenerable
files were moved aside, not deleted), and `python -m regime.run` was then run
twice with no `--pull`, each rebuilding sections 1 to 7 from the pinned
snapshots in `data/raw/`.

`outputs/tables/runtime.csv`:

```
run_started,section,seconds
20260923T105636Z,1,3.332
20260923T105636Z,2,1.984
20260923T105636Z,3,238.441
20260923T105636Z,4,4.538
20260923T105636Z,5,26.724
20260923T105636Z,6,517.234
20260923T105636Z,7,3.119
20260923T111100Z,1,3.452
20260923T111100Z,2,1.621
20260923T111100Z,3,205.393
20260923T111100Z,4,3.283
20260923T111100Z,5,21.309
20260923T111100Z,6,475.091
20260923T111100Z,7,2.651
```

Run 1 total 795.37 s (13 m 15 s); run 2 total 712.80 s (11 m 53 s). **No
section came near its threshold**: sections 3 and 6 are long-step sections at
60 minutes and reached 3.97 and 8.62 minutes; every other section is under 30
seconds against a 20-minute threshold. Nothing was reduced — `hmm.n_restarts`
is the configured 20, `bootstrap.n_replications` the configured 2000, and every
cell of every grid ran.

### Step 7.3 — the comparison table

`compare_output_trees(run1_snapshot, outputs, cfg)` over every file under
`outputs/`, excluding `outputs/tables/runtime.csv` only:

```
compared_as
bytes     498
pixels      5

identical 503 of 503
```

**Zero files differ.** The five PNG rows, compared by
`np.array_equal(mpimg.imread(a), mpimg.imread(b))`:

```
                                  path compared_as                                                         sha_run1                                                         sha_run2  identical
charts/conditional_sharpe_filtered.png      pixels dcece898284dabcb718143cbe31f1a70077255b2459afa4404214c04b3417592 dcece898284dabcb718143cbe31f1a70077255b2459afa4404214c04b3417592       True
charts/conditional_sharpe_smoothed.png      pixels 2b61ab293bbe0ebb11ebe0d6abf0b53b6fa69d9331f91a0d7c21dac921ec5fdd 2b61ab293bbe0ebb11ebe0d6abf0b53b6fa69d9331f91a0d7c21dac921ec5fdd       True
            charts/features_review.png      pixels ac8e21525b57484164e2b2996b062196a9d67b15c65157d0967ee7a97539aa76 ac8e21525b57484164e2b2996b062196a9d67b15c65157d0967ee7a97539aa76       True
           charts/regimes_timeline.png      pixels 22371b9dd0ce40d602732c562756aa7bf9ccd7adc2a62b75d0427bff6a4c4a6f 22371b9dd0ce40d602732c562756aa7bf9ccd7adc2a62b75d0427bff6a4c4a6f       True
            charts/timed_vs_static.png      pixels af6f3589ec2ff3dadafdaa8d64a25131bf1ad1ca0f815052b3c62155f2d49fc3 af6f3589ec2ff3dadafdaa8d64a25131bf1ad1ca0f815052b3c62155f2d49fc3       True
```

The `sha_run1` and `sha_run2` columns are printed for the PNGs as well, and
they happen to agree — the `metadata={"Software": None}` argument is doing its
job and the PNG bytes are reproducible too — but the **comparison** for a PNG
is the decoded pixel array, because a PNG carries chunk-level metadata that no
`savefig` argument is guaranteed to suppress on every matplotlib version.

The first ten CSV rows, as a sample of the 498 compared by bytes:

```
                                           path compared_as                                                         sha_run1                                                         sha_run2  identical
                                tables/.gitkeep       bytes e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855       True
                        tables/anchor_chain.csv       bytes 6e3f21a77452e0acd7c8c6e598a9154dddc92af3170517d20412c7887845787e 6e3f21a77452e0acd7c8c6e598a9154dddc92af3170517d20412c7887845787e       True
                            tables/bic_by_k.csv       bytes 191505af0f19bacdfb4a76e8dd4d4db13da3cb7f4513e78caabbf234f7d2502d 191505af0f19bacdfb4a76e8dd4d4db13da3cb7f4513e78caabbf234f7d2502d       True
          tables/bootstrap_nan_replications.csv       bytes 64692d6981a4eed3162bb110a10d5cf30c98152968d6c9a246bc82dffcef0f6e 64692d6981a4eed3162bb110a10d5cf30c98152968d6c9a246bc82dffcef0f6e       True
                       tables/check_outputs.csv       bytes 413566d8210dc0886b21b6dc095d270c506cbbbb95728f1f153c856f4c1c7b4d 413566d8210dc0886b21b6dc095d270c506cbbbb95728f1f153c856f4c1c7b4d       True
              tables/classifier_diagnostics.csv       bytes f2f0ac2bdd9cb2a825d533edb7dec555b55ba1bcf8f005c82a3f1f46d4b24fac f2f0ac2bdd9cb2a825d533edb7dec555b55ba1bcf8f005c82a3f1f46d4b24fac       True
tables/conditional_differences_gmm_filtered.csv       bytes 7fa20418bd0736da6901c27b9e58ed85b763eb4a6d599bf6360ec52698068db9 7fa20418bd0736da6901c27b9e58ed85b763eb4a6d599bf6360ec52698068db9       True
tables/conditional_differences_hmm_filtered.csv       bytes eb762390920fbbec75c559c1f8e2b52fc660d84aaedc7d2448ab0a65fe5bfb1a eb762390920fbbec75c559c1f8e2b52fc660d84aaedc7d2448ab0a65fe5bfb1a       True
tables/conditional_differences_hmm_smoothed.csv       bytes 35d139e0f98b34549a418cfdfcd5a5841913019e8ca7c3456f62d80dbcf0ca43 35d139e0f98b34549a418cfdfcd5a5841913019e8ca7c3456f62d80dbcf0ca43       True
       tables/conditional_differences_rules.csv       bytes 4fc0096cb0a8684924fed2c2984bf494ef8a5684b4de379ed896dd3fd6edcf9d 4fc0096cb0a8684924fed2c2984bf494ef8a5684b4de379ed896dd3fd6edcf9d       True
```

and the last four:

```
                                   path compared_as                                                         sha_run1                                                         sha_run2  identical
tables/transition_matrix_2024-12-31.csv       bytes 0929e2b65cf252242c2ded0de2e02f1e8c340c1dd2cd7b35d25a7a69b52ee127 0929e2b65cf252242c2ded0de2e02f1e8c340c1dd2cd7b35d25a7a69b52ee127       True
tables/transition_matrix_2025-12-31.csv       bytes cad2f808a3d145c55bcc247ba55a2f049999c683303ad78a590b0bfc12a1abdc cad2f808a3d145c55bcc247ba55a2f049999c683303ad78a590b0bfc12a1abdc       True
         tables/unconditional_stats.csv       bytes 604b1cff8120b76d95fbd6be0b8a3c2d55d35933bcf11b071af95046202a0534 604b1cff8120b76d95fbd6be0b8a3c2d55d35933bcf11b071af95046202a0534       True
            tables/weight_deviation.csv       bytes 01a581ace26d834f79e328e257e3ad0e517f6f9e063043245eba5e94f7069e74 01a581ace26d834f79e328e257e3ad0e517f6f9e063043245eba5e94f7069e74       True
```

The full 503-row table was written to the session scratchpad rather than into
`outputs/`, because a comparison of two runs is not an output of either run and
committing it would make the next run's comparison non-idempotent.

**This is a same-machine property and is stated as one.** Both runs are the
same interpreter, the same NumPy and the same BLAS on one Windows 11 machine.
Floating-point output moves at roughly the 1e-12 level across Python, NumPy,
scipy and BLAS versions, and a CSV written at full repr precision turns a
1e-12 difference into different bytes. **Nothing here supports a claim that a
different environment reproduces these bytes**; what it supports is that the
pipeline carries no hidden state, no wall-clock, no unseeded randomness and no
dependence on what was already on disk. The README says the same in its
methods note.

### A third, independent check on the same property

`git status --short` after run 1 showed only the new untracked
`outputs/tables/runtime.csv`:

```
$ git status --short
 M regime/run.py
 M tests/test_run.py
?? outputs/tables/runtime.csv
```

The two modified files are step 7.3's own source changes. **No committed output
file was modified**, so run 1 — starting from an empty `data/processed/` —
reproduced every output committed by sections 1 to 6 and by steps 7.1 and 7.2,
byte for byte. That is a stronger statement than run-1-against-run-2, because
those earlier outputs were written on a different day from a differently
populated `data/processed/`.

### Step 7.4 — every number in the README, and the table it comes from

Sixteen fenced blocks in `README.md` are verbatim renderings of committed
tables. Each was checked by re-rendering the table with
`pd.read_csv(...).to_string(index=False)` and asserting string equality against
the block in the file:

```
OK   bic          outputs/tables/bic_by_k.csv (all 4 rows)
OK   drift        outputs/tables/param_drift.csv (state 2, indpro_chg12, the 2 refits)
OK   diff_filt    outputs/tables/conditional_differences_hmm_filtered.csv (excludes_zero)
OK   diff_smooth  outputs/tables/conditional_differences_hmm_smoothed.csv (excludes_zero)
OK   stats_filt   outputs/tables/conditional_stats_hmm_filtered.csv (all 18 rows)
OK   uncond       outputs/tables/unconditional_stats.csv (all 6 rows)
OK   splitdiff    outputs/tables/conditional_refit_split_differences.csv (excludes_zero)
OK   decomp       outputs/tables/timing_headline_decomposition.csv (all 4 rows)
OK   gap          outputs/tables/filtered_smoothed_gap.csv (the 1 cell excluding zero)
OK   robust       outputs/tables/robustness/robustness_summary.csv (all 6 rows)
OK   frag         outputs/tables/robustness/fragility_summary.csv (both rows)
OK   diag         outputs/tables/classifier_diagnostics.csv (both rows)
OK   kvar         outputs/tables/robustness/k_variant_outcomes.csv (all 4 rows)
OK   headline     outputs/tables/timing_results.csv (the headline cell)
OK   four         outputs/tables/timing_results.csv (eta 0.5, hmm_filtered, 10/20 bp, lag 0/1)
OK   smooth4      outputs/tables/timing_results.csv (eta 0.5, hmm_smoothed, 10/20 bp, lag 0/1)
```

The `kvar` block is the one block indented two spaces, because it sits inside a
list item; it is verbatim after removing that uniform indent. Every other block
is at column 0 and matches exactly.

The derived numbers in the prose — the ones that are not a cell of a table —
and where each comes from:

| number in the README | how it is derived | from |
|---|---|---|
| 9 of 20 label changes on refit dates, 45% | the two columns of one row | `classifier_diagnostics.csv`, `core` row |
| 1 of 18 pairwise differences exclude zero | row count of `excludes_zero` | `conditional_differences_hmm_filtered.csv` |
| 3 of 18, smoothed | the same | `conditional_differences_hmm_smoothed.csv` |
| 6 of 18 refit-split differences | the same | `conditional_refit_split_differences.csv` |
| 0 of 72 cells with `p_one_sided` < 0.10; smallest 0.178 | count and min of one column | `timing_results.csv` |
| 20 of 72 cells with `diff` > 0 | count of one column | `timing_results.csv` |
| 7 of 72 intervals exclude zero, all `rules`, all below | count of `(diff_p05 > 0) \| (diff_p95 < 0)` and the `source` of the 7 | `timing_results.csv` |
| `diff` range −0.381 to +0.073 | min and max of one column | `timing_results.csv` |
| 1.88 z max parameter drift | the difference of two quoted rows, −2.328620 and −0.451311 | `param_drift.csv` |
| 45 BIC points, K = 3 over K = 4 | 2943.099433 − 2897.822739 | `bic_by_k.csv` |
| 18 of 66 matched above 1 z, max 2.689 | count and max of `matched_distance` | `anchor_chain.csv` |
| 48 of 66 sort disagreements | count of `state != sort_rule_state` | `anchor_chain.csv` |
| 20 of 51, max 3.80 (10feat) | the same two, on the variant's file | `robustness/10feat/anchor_chain.csv` |
| mean gap −0.034 over 18 cells | mean of the `gap` column | `filtered_smoothed_gap.csv` |
| 440 of 440 HMM and 440 of 440 GMM restarts converged | 22 files × 20 rows, `converged` all True | `hmm_restarts_<refit>.csv`, `gmm_restarts_<refit>.csv` |
| 72 of 258 months (27.9%) in fallback | `n_fallback` and `fallback_share`, `min_regime_obs` 24, `hmm_filtered` | `robustness/minobs_fallback.csv` |
| 14.0% at `min_regime_obs` 12 | `fallback_share` of the 12 rows | `robustness/minobs_fallback.csv` |
| 91 / 61 / 106 and 73 / 38 / 147 months per state | the `n` column | `conditional_stats_hmm_filtered.csv`, `..._hmm_smoothed.csv` |
| NaN counts per feature (168, 12, 12, 12, 0…) | sum of `n_nan` over years | `feature_sanity.csv` |
| 438 of 440 decision dates at t−1, 99.55% | quoted from the section 1 evidence | `review/section_1.md`, step 1.4 |
| 68.2% filtered/smoothed agreement | one cell | `classifier_diagnostics.csv`, `core` row |
| `p_one_sided` moves 0.011 across block sizes | max − min of one column | `timing_results_blocksize.csv` |

The raw rows behind the four that are not a single cell:

`timing_results.csv`, the 7 cells whose interval excludes zero — every one a
`rules` cell, every one with `diff_p95 < 0`:

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

`filtered_smoothed_gap.csv` in full, the 18 rows the mean of −0.03351820 is
taken over:

```
factor  state       gap   gap_p05   gap_p95
Mkt-RF      0 -0.135670 -1.238004  0.754707
Mkt-RF      1  0.360084 -0.284289  1.026063
Mkt-RF      2  0.037677 -0.586075  0.672053
   SMB      0 -0.339167 -1.066862  0.353638
   SMB      1 -0.121899 -0.554719  0.297752
   SMB      2  0.066917 -0.388635  0.454746
   HML      0 -0.230605 -1.024985  0.533849
   HML      1  0.005437 -0.380035  0.623029
   HML      2  0.126071 -0.274896  0.599473
   RMW      0 -1.065696 -1.900999 -0.294969
   RMW      1  0.429990 -0.318265  1.295200
   RMW      2  0.248040 -0.071779  0.702799
   CMA      0  0.300504 -0.439539  1.035691
   CMA      1  0.011459 -0.662595  0.661643
   CMA      2 -0.324943 -0.746757  0.127955
   UMD      0 -0.127134 -0.654867  0.482525
   UMD      1 -0.036317 -0.477775  0.671045
   UMD      2  0.191927 -0.134686  0.515041
```

`robustness/minobs_fallback.csv`, the `hmm_filtered` rows the 27.9% and 14.0%
come from:

```
 min_regime_obs       source  eta  n_months  n_fallback  fallback_share  mean_abs_deviation
             12 hmm_filtered 0.25       258          36        0.139535            0.045640
             12 hmm_filtered 0.50       258          36        0.139535            0.091279
             12 hmm_filtered 1.00       258          36        0.139535            0.182559
             24 hmm_filtered 0.25       258          72        0.279070            0.038877
             24 hmm_filtered 0.50       258          72        0.279070            0.077755
             24 hmm_filtered 1.00       258          72        0.279070            0.155510
             36 hmm_filtered 0.25       258         108        0.418605            0.030477
             36 hmm_filtered 0.50       258         108        0.418605            0.060953
             36 hmm_filtered 1.00       258         108        0.418605            0.121906
```

`timing_results_blocksize.csv` in full, the three `p_one_sided` values the
0.011 spread comes from (0.5975 − 0.5865):

```
 block_size  eta  lag  cost_bp       source  sharpe_static  sharpe_timed   diff  diff_p05  diff_p95  p_one_sided  mean_turnover  n_months
          3  0.5    1       20 hmm_filtered       0.206378      0.179378 -0.027 -0.215921  0.156870       0.5975       0.043992       257
          6  0.5    1       20 hmm_filtered       0.206378      0.179378 -0.027 -0.227855  0.170963       0.5865       0.043992       257
         12  0.5    1       20 hmm_filtered       0.206378      0.179378 -0.027 -0.224503  0.171390       0.5955       0.043992       257
```

`feature_sanity.csv`, `n_nan` summed by feature over all years:

```
feature
breakeven_chg12    168
cpi_3m_ann           1
dgs10_chg12         12
dgs10_level          0
dollar_chg12        12
indpro_chg12         0
log_vix              0
oil_chg12           12
slope_2s10s          0
unrate_chg12         0
```

`cpi_3m_ann`'s single NaN is the 2026-02-28 row the README names as the one
dropped decision date; the three 12s are the first 12-row change of each
market series; `breakeven_chg12`'s 168 is T10YIE starting in 2003.

### The test that guards the README

`tests/test_readme.py::test_readme_headline_numbers_match_outputs` parses every
fenced block in `README.md` whose header carries both the `timing_results`
column names and `p_one_sided`, splits each row on whitespace, looks the cell
up in `timing_results.csv` by `(eta, lag, cost_bp, source)` and compares seven
float columns to 5e-7 and `n_months` exactly. On the current README that is 13
quoted rows across 3 blocks, and it asserts the headline cell appears at least
twice. Two further tests assert the null is in the first paragraph ahead of any
question heading, and that the trimmed +0.044294 is accompanied by the sentence
naming it a diagnostic rather than a result.

## Tests run

Exact command and full output, unedited, at the end of the section:

```
$ pytest -q
........................................................................ [ 55%]
.........................................................                [100%]
129 passed in 22.00s
```

`pytest -q` also passed at the end of every step. No test was deleted, skipped,
marked xfail or loosened.

| after step | `pytest -q` | tests added by the step |
|---|---|---|
| 7.1 | 119 passed | 3 (`test_charts.py`) |
| 7.2 | 121 passed | 2 (`test_tables.py`) |
| 7.3 | 129 passed | 5 (`test_run.py`) and 3 (`test_readme.py`) |
| 7.4 | 129 passed | 0 further |

One thing to read correctly in that table: `tests/test_readme.py` (3 tests)
belongs to step 7.4 and is in the 7.4 commit, but it was written before the 7.3
commit was made, so its 3 tests are already inside the 129 counted at 7.3. The
commits are correct — `e55b228` contains only `regime/run.py`,
`tests/test_run.py` and `runtime.csv` — and only the intermediate count is
affected.

## Runtime per step

| step | wall-clock | machine | notes |
|---|---|---|---|
| 7.1 | 0:06 | Windows 11, Python 3.11 | `write_charts` end to end, four PNGs at 300 dpi |
| 7.2 | 0:03 | " | `write_regime_labels` + `check_outputs` |
| 7.3 | 25:08 | " | two full runs: 795.37 s + 712.80 s, plus the comparison |
| 7.4 | — | " | prose; the test runs in under a second |

Step 7.3's threshold is 60 minutes (the instruction message set it there
because it reruns every section); it took 25 minutes 8 seconds. Every other
step is under 20 minutes by three orders of magnitude. **No step was over its
threshold and nothing was reduced to fit one** (rule 10).

Per-section runtimes inside step 7.3 are in `runtime.csv` above; the longest
single section is section 6 at 517.2 s against its 60-minute threshold.

## Not verified

- **Cross-environment reproducibility.** The idempotence table is two runs of
  one interpreter on one machine. Nothing here tests another Python version,
  another NumPy or BLAS, or another operating system, and the review does not
  claim it. A 1e-12 floating-point difference would break byte equality on the
  CSVs and would likely survive the pixel comparison on the PNGs.
- **That a fresh clone reproduces the tables without a FRED key.** The README
  states it, and it follows from convention 14 (raw committed) plus the fact
  that run 1 rebuilt everything from `data/raw/` with an empty
  `data/processed/` — but no actual fresh clone into a new directory was made
  and run in this session. The strong form of the claim is untested.
- **The charts' visual correctness beyond the assertions.** The tests assert a
  non-empty PNG, the border count and placement, and the legend strings. Nobody
  asserted that the shading spans line up with the label dates or that the twin
  axes are on the right series; that was checked by eye on
  `regimes_timeline.png` and `conditional_sharpe_filtered.png` and not by test.
- **`check_outputs` on a malformed table.** The test covers absent, present
  and wrong-columns. It does not cover a file that is present with the right
  header and no rows, which would report `columns_ok` True.
- **The prose of the README.** Only numbers are tested. No test asserts that a
  sentence describing a number describes it correctly; the derivation table
  above is the manual check.
- **The 2026-02-28 dropped row's effect on anything downstream.** It is
  reported in three places and was not traced through the conditional or timing
  numbers.
- **The `hmm_filtered_assigned` column is True on every date it is available.**
  Verified and explained above, but not asserted by a test on the real data —
  the step 7.2 test asserts the *coverage* rule on synthetic labels with a
  deliberately mixed `assigned` flag.

## Open questions

**None.** No item was appended to `decisions/OPEN.md` this section, and no step
was left incomplete. Item 1 — the only item in the file — was marked resolved
under the reviewer's Option A in `c7f99ae` and kept in place.

## Files changed

**`07: instructions`** (`c4f05b3`)
- added `instructions/07_section_7.md`

**`section 7: reviewer decisions`** (`c7f99ae`)
- added `decisions/section_6_review.md`
- modified `decisions/OPEN.md` (item 1 headed RESOLVED, resolution appended; nothing deleted)

**`step 7.1`** (`d9f3c26`)
- modified `regime/charts.py` (+`STATE_COLOURS`, `_last_refit_rows`, `state_legend_labels`, `_shade_by_label`, `regimes_timeline`, `conditional_sharpe_heatmap`, `timed_vs_static`)
- modified `regime/run.py` (+`write_charts`, `_chart_legend`, `section_7`; `section_7` replaces `_not_built(7)`)
- modified `tests/test_charts.py` (+3 tests, +2 fixtures)
- modified `tests/test_run.py` (`test_unbuilt_section_raises` now exercises `_not_built(8)`)
- added `outputs/charts/regimes_timeline.png`, `conditional_sharpe_filtered.png`, `conditional_sharpe_smoothed.png`, `timed_vs_static.png`

**`step 7.2`** (`508f8c1`)
- modified `regime/tables.py` (+`REGIME_LABEL_COLUMNS`, `write_regime_labels`, `SECTION_8_TABLES`, `CHECK_OUTPUTS_COLUMNS`, `check_outputs`)
- modified `regime/run.py` (+`write_published_tables`, called from `section_7`)
- modified `tests/test_tables.py` (+2 tests, +1 fixture)
- added `outputs/regimes/regime_labels.csv`, `outputs/tables/check_outputs.csv`

**`step 7.3`** (`e55b228`)
- modified `regime/run.py` (+`RUNTIME_COLUMNS`, `append_runtime`, `COMPARISON_COLUMNS`, `compare_output_trees`; `main` now times each section and appends a row)
- modified `tests/test_run.py` (+5 tests)
- added `outputs/tables/runtime.csv`

**`step 7.4`** (`9c21f1d`)
- modified `README.md` (the whole Results section; "Results pending" removed)
- added `tests/test_readme.py` (3 tests)

No file was deleted. `data/processed/` was emptied for step 7.3 and rebuilt by
the run; it is gitignored (convention 14) and appears in no commit.

## Reviewer reads

1. **`README.md`** — the deliverable. Check that the first paragraph states the
   null and that a reader who stops there has the result; check that no number
   appears without the file it came from; check that the timing result is
   stated as indistinguishable from zero in both directions and that the
   trimmed +0.044 is never presented as a result.
2. **`decisions/section_6_review.md` and `decisions/OPEN.md`** — that the
   reviewer's four answers are recorded as given, and that item 1 is marked
   resolved under Option A and kept rather than deleted.
3. **The "Step 7.3 — the comparison table" section above** — that the
   same-machine caveat is stated and that the claim made is no stronger than
   503 of 503 files identical across two runs of one interpreter.
4. **`regime/tables.py::check_outputs` and `SECTION_8_TABLES`** — that the set
   of tables checked is the kickoff's section 8 list and that the boundary of
   what it does not cover is stated.
5. **`regime/charts.py::conditional_sharpe_heatmap`** — that the border is
   driven by `excludes_zero` and nothing else, and that the docstring says what
   a border does and does not mean.
6. **`tests/test_readme.py`** — that it would actually fail if a quoted number
   drifted from `timing_results.csv`.
7. **The "Not verified" section above** — in particular that no fresh clone was
   made and run, though the README asserts a fresh clone works.
