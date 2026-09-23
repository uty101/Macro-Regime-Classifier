# Status — 06, section 6

**Outcome: completed.** The reviewer's three answers, the new step 6.7, and
every step of Section 6 of `PLAN.md` (6.1 to 6.7) were built in one session,
without pausing between steps. The session did not stop early and hit no rule
4 threshold. It refused one decision under rule 1 — `decisions/OPEN.md` item
1 — and finished every step that did not depend on it, which was all of them.

## Step reached

All of 6.1 to 6.7, plus the two commits the instruction message required
before them.

| step | what | commit |
|---|---|---|
| — | this instruction message, saved verbatim (rule 11) | `dc74643` |
| — | the reviewer's three answers and the PLAN.md changes they make | `722ad1d` |
| 6.1 | K ∈ {2, 3, 4, 5}, and the one K that cannot be filtered | `9c9adc2` |
| 6.2 | the diagonal-covariance rerun, and the `covars_` round trip it needed | `276cf8c` |
| 6.3 | the ten-feature run, from 2004-01-31 with a 2009-12-31 first refit | `e899d30` |
| 6.4 | `min_regime_obs` over 12, 24, 36, with the fallback share beside the diff | `4524c3c` |
| 6.5 | bootstrap `block_size` over 3, 6, 12, headline cell only | `4f68590` |
| 6.6 | the non-primary feature set's conditional statistics, bootstrap and grid | `86281f1` |
| 6.7 | how much of each headline statistic one month is carrying | `a47a3c3` |
| — | `review/section_6.md` | `20c8c03` |

`pytest -q`: **116 passed**, at the end of every step and at the end of the
section (104, 106, 108, 110, 111, 112, 116 after 6.1 to 6.7 respectively). No
test was deleted, skipped, marked xfail or loosened.

## The project's result is unchanged

The headline cell is where `config.toml` fixed it before section 5 ran — η
0.5, lag 1, 20 bp, `hmm_filtered`:

```
 eta  lag  cost_bp       source  sharpe_static  sharpe_timed   diff  diff_p05  diff_p95  p_one_sided  mean_turnover  n_months
 0.5    1       20 hmm_filtered       0.206378      0.179378 -0.027 -0.227855  0.170963       0.5865       0.043992       257
```

**Nothing in section 6 replaces it and no sentence here or in
`review/section_6.md` presents a robustness variant as the project's result.**
Step 6.1's `k3` variant reproduces that row to every digit printed, which is
the check that the variant code path and the main code path are the same code.

## The six variants that completed

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

## The sub-0.1 check was not triggered

**No variant produced a headline-equivalent `diff` with `p_one_sided` below
0.1, and no variant produced a positive `diff` at all.** Ascending:

```
variant      diff  p_one_sided
nolevel -0.001413       0.5125
   diag -0.007525       0.5185
     k3 -0.027000       0.5865
 10feat -0.083143       0.7260
     k2 -0.095716       0.7695
     k4 -0.102487       0.8180
```

The instruction message said a single variant clearing 0.1 out of roughly ten
is what chance produces. Zero of six cleared it, and the lowest is 0.5125 —
the bootstrap's replications split almost evenly on the sign for every
variant, as they did for the main run.

The three conditions the check names were verified anyway, because every
variant runs through the same three functions and a fault in any of them would
have been a fault in the main run too:

- **Weights use only `s < t`.** Every variant calls `regime.strategy.weights`,
  which calls `trailing_conditional_sharpe`, which selects `joined.index < t`
  strictly. `tests/test_strategy.py::test_trailing_sharpe_is_strictly_before_t`
  plants a 1e3 return in month t+1 and in month t+2 and asserts `S` and `n_k`
  do not move. Section 6 adds no second path to a weight — `variant_timing_grid`
  and `run_robustness_blocksize` both go through it.
- **The lag is applied as specified.** Every variant backtests through
  `regime.strategy.backtest`, whose shift is positional on an index
  `join_next_return` has already asserted to be a complete monthly month-end
  sequence. `test_backtest_three_month_literal` checks PLAN.md step 5.3's
  example at both lags with every number written out.
- **The static comparator runs on identical earning months.** `timing_cells`
  asserts `timed.index.equals(static.index)` per cell and raises otherwise;
  `run_robustness_blocksize` and `headline_net_returns` repeat that assertion
  for the single cell each builds. Every variant's `sharpe_static` is
  **0.206378** — literally one number — except `10feat`, whose out-of-sample
  window genuinely starts at its own first refit and whose `n_months` of 197
  against 257 says so on the same row.

## What section 6 adds to the picture

1. **The null does not depend on K.** K = 2, 3 and 4 give `diff` of −0.0957,
   −0.0270 and −0.1025, `p_one_sided` 0.7695, 0.5865 and 0.8180. K = 3, the
   chosen one, is the least bad of the three.
2. **Nor on the covariance structure, nor on the feature set.** `diag` gives
   −0.0075 at p 0.5185 and `nolevel` −0.0014 at p 0.5125. Both come nearer to
   breaking even than the main run and neither is distinguishable from zero.
3. **Nor on the bootstrap's block size.** Over a factor of four in block size,
   `p_one_sided` moves 0.011, from 0.5975 to 0.5865 to 0.5955, and the same 4
   of 18 conditional cells exclude zero at every block size. The width of the
   interval is not what is keeping the answer indistinguishable from zero.
4. **The `min_regime_obs` effect is mechanical, and the new columns show it.**
   Going 24 → 12 improves `diff` from −0.0270 to −0.0220 while the fallback
   share halves (27.9% → 14.0%), the mean absolute weight deviation rises
   (0.0778 → 0.0913) and the mean turnover rises (0.0440 → 0.0547). More
   trading, not more signal, exactly as the reviewer's Q2 answer required it be
   read. `p_one_sided` stays at 0.57.
5. **The headline `diff` is carried by one month; the UMD pair is not.**
   `fragility_summary.csv`:

   ```
             statistic  full_value   loo_min   loo_max  n_sign_flips  share_sign_flips  trimmed_value  n_months
         headline_diff   -0.027000 -0.049947  0.012667             1          0.003891       0.044294       257
   umd_0_2_sharpe_diff   -1.024094 -1.156142 -0.822955             0          0.000000      -1.209043       197
   ```

   One removal of 257 flips the headline's sign — 2026-07-31, the last month
   of the sample and the largest timed-minus-static month at −3.77% — and the
   5% trim moves it to +0.0443. The UMD 0-against-2 difference flips on no
   single month, and the trim leaves it at −1.209, more negative than it
   started. 2009-03-31 is indeed its most influential month (a −34.36% UMD
   return in state 2, worth +0.201 on removal), but it is not carrying the
   statistic.

   **The trimmed +0.0443 is a fragility diagnostic, not a result.** It carries
   no interval, it is not a cell of any grid, and it is not what the project
   found. The project found `diff = −0.027`, `p_one_sided = 0.5865`.

## Questions and blockers for the reviewer

### 1. `decisions/OPEN.md` item 1 — K = 5 cannot complete the expanding protocol

This is the one thing that blocked work, and it blocked exactly one variant.

`run_expanding_hmm` at K = 5 on the primary d = 8 set fails at the
**2006-12-31** refit. hmmlearn fits it; the fitted covariance of one state is
numerically singular and `forward_filter` refuses it. The cause is in the
variant's own `state_counts.csv` — the only degenerate row in 110 across all
four K values:

```
refit_date  state  n_rows  degenerate
2006-12-31      4       7        True
```

Seven training rows in eight dimensions, so the covariance cannot have rank
above 7. Its eigenvalues at that refit are

```
2.170e-16  2.506e-03  7.736e-03  8.326e-02  1.292e-01  7.726e-01  1.629e+00  3.379e+00
```

— condition number 1.6e16, on a matrix symmetric to 2.2e-16, so this is rank
deficiency and not a symmetry artefact. The model has `n_params(5, 8) = 240`
free parameters and that refit trains on 192 rows. `hmm.min_covar` (1e-3)
floors the diagonal and cannot prevent a rank deficiency in a full covariance.

`select_k` completes at K = 5 because it fits only the 168-row first window
and never filters forward with those parameters, which is why section 3 did
not see this.

`PLAN.md` step 6.1 says "for each K in `cfg.hmm_k_grid`" and says nothing
about a K that cannot be filtered, so the session did not choose. The two
options are in `decisions/OPEN.md`:

- **A** — report K = 5 as a failure of the variant and leave the code alone.
  Cost: the robustness of the K choice is tested over three values, not four.
- **B** — pass `allow_singular=True` in
  `regime/models/hmm_numpy.py::log_emissions`. Cost: this is a change to the
  **main run's** filter, not to a variant. K = 3's covariances are well
  conditioned so its numbers would not move, but the filter would stop
  refusing a degenerate state anywhere, and a density evaluated on a
  rank-deficient covariance is not comparable across states.

What the session did: neither. K ∈ {2, 3, 4} are complete and reported; K = 5
is recorded in `outputs/tables/robustness/k_variant_outcomes.csv` with its
parameter count and in `review/section_6.md`. **No step was left incomplete
because of it.**

### 2. Two definitions step 6.7 fixed by implementation, which the reviewer may want to set

The instruction message fixed the statistics but not these two details. Both
are documented at their point of definition in `regime/robustness.py` and both
are visible in the outputs, but neither was the reviewer's choice.

- **The trim count is `floor(0.05 n)`.** At n = 257 that removes 12 months
  (4.67%), not 13 (5.06%); at n = 197 it removes 9. Floor rather than round,
  so "the 5% of months" never removes more than 5% of them. Rounding up would
  make the headline's trimmed value slightly different; the direction of the
  finding would not change.
- **The pairwise trim measures influence differently from the headline trim.**
  For the headline there is a timed and a static return in the *same* month, so
  "the largest absolute timed-minus-static difference" is a per-month quantity
  and was used as specified. For a pairwise state difference the two states
  occupy **disjoint** months and no such pairing exists, so a month's influence
  is taken to be how far its removal moves the statistic, `loo_diff −
  full_diff` — which is what the leave-one-out column already measures. If the
  reviewer wants a different definition (for example the deviation of the
  month's return from its own state's mean), it is a one-function change in
  `trimmed_pairwise_diff`.

### 3. `PLAN.md` step 6.2's claim about `covars_` is half right, and the fix touches main-run code

Step 6.2 says `HMMParams.covars` can stay (K, d, d) "because hmmlearn's
`covars_` property returns full-shaped matrices for every covariance type, so
`forward_filter` and `anchor` are unchanged". That is true of the **getter**
and of `forward_filter` and `anchor`. It is false of the **setter**, which
validates against the covariance type's internal shape and rejects (K, d, d)
for `"diag"`. `model_from_params` raised the first time the diagonal variant
called `predict_proba`.

The fix, `regime/models/hmm.py::covars_for_setter`, does the conversion in one
place and checks the off-diagonals are zero before discarding them. `"full"`
returns its input unchanged, so no main-run output moves — and none did: the
`k3` variant still reproduces the main headline row exactly, and `section_6`
rerun end to end left `git status` clean. The reviewer should know that a
function the published K = 3 classifier uses was touched, even though its
behaviour on that path is identical.

### 4. Two things section 6 found that are not about timing

Neither is a blocker; both belong in step 7.4's "What did not work".

- **The `10feat` variant's states are the least stable in the project.** 20 of
  its 51 chained states matched at more than 1 z, against a maximum of 3.80
  (`outputs/tables/robustness/10feat/anchor_chain.csv`). No step 6.1 run comes
  close.
- **The anchor-versus-sort disagreement grows with K**: 6 of 44 rows at K = 2,
  48 of 66 at K = 3, 58 of 88 at K = 4, 82 of 110 at K = 5. Convention 16's
  premise — that the sort was ordering noise — holds harder the more states
  there are.

Nothing else blocked the work.

## Rule 10 — runtime

No step came close to its threshold. Nothing was reduced: `hmm.n_restarts` is
the configured 20, `bootstrap.n_replications` the configured 2000, and every
cell of every grid was run.

```
step                          seconds   threshold
6.1 (K in 2, 3, 4, 5)           339.3      3600
6.2 (diag)                       52.0      3600
6.3 (10 features)                32.1      3600
6.4 (min_regime_obs x 3)         55.1      1200
6.5 (block_size x 3)              3.3      1200
6.6 (non-primary feature set)     9.5      3600
6.7 (fragility)                   1.7      1200
section 6 end to end            455.5         —
```

**Step 6.7 was measured before being classified, as the instruction required:
1.7 seconds, so it is not added to `run.long_steps` and `config.toml` is
unchanged.**

Step 6.1's 339 s includes ~95 s spent on the K = 5 attempt before the filter
refused it. Step 6.6 is cheap (9.5 s) because section 3 already fits the
non-primary feature set on every run and step 6.6 adds only the three things
section 3 does not do.

## Section 6 is idempotent

`section_6(cfg)` was run end to end after all seven steps were committed.
`git status --short` afterwards printed nothing: every file under `outputs/`
was rewritten byte-identically.

## Deviations from the letter of the specification

All additive, all flagged here rather than left to be found.

- **`regime/tables.py::changes_on_refit_dates` extracted** from
  `classifier_diagnostics_row`, which now calls it, so section 6's
  `n_changes_on_refit_dates` and section 3's come from one definition. The
  diagnostics row's numbers are unchanged.
- **`regime/models/hmm.py::covars_for_setter` added**; see question 3 above.
- **`outputs/tables/robustness/k_variant_outcomes.csv` is an extra table**, not
  named in `PLAN.md`. It exists so a K that cannot complete is an output rather
  than a swallowed exception, and
  `test_k_variant_outcomes_cover_every_k_in_the_grid` asserts every K of the
  grid appears in it with a status and, where it failed, a reason.
- **`run_robustness_k` catches `numpy.linalg.LinAlgError` specifically** — not
  a broad `except` — records the outcome and continues to the next K. Catching
  nothing would have stopped the section at 6.1 with five steps unbuilt;
  catching more would have hidden a different fault.
- **Seven tests beyond those `PLAN.md` and the instruction message list:**
  `test_robustness_cfg_overrides_are_applied_last`,
  `test_k_variant_outcomes_cover_every_k_in_the_grid`,
  `test_diag_covars_round_trip_through_the_hmmlearn_setter`,
  `test_minobs_fallback_share_falls_as_min_regime_obs_falls` (required by the
  PLAN.md amendment the reviewer's Q2 answer makes), and
  `test_pairwise_loo_covers_the_months_of_both_states`. The round-trip test
  exists because the bug it covers was found by a crash, not by a test.
- **`regime/robustness.py` carries two functions the instruction did not name**
  — `trimmed_pairwise_diff` and `trimmed_count` — because the pairwise trim
  needs its own influence measure (question 2) and because the count a trim
  removes is what one of the three required tests asserts.
- **Step 6.7 finds its pairwise target by reading
  `conditional_differences_hmm_filtered.csv`** rather than hard-coding
  `(UMD, 0, 2)`, so it cannot go on testing the fragility of a difference
  section 4 has stopped finding. On the current outputs it resolves to exactly
  `(UMD, 0, 2)`, and the output file is named
  `leave_one_month_out_umd_0_2.csv` as `PLAN.md` specifies.
- **`tests/test_run.py::test_unbuilt_section_raises`** asserted that section 6
  was not built; it now asserts it of section 7. One line, in the step 6.1
  commit.
- **`_variant_cfg` is kept** as a one-line wrapper over the new
  `robustness_cfg`, so section 3's call site is untouched.

## Final remote state

```
$ git log origin/main --oneline -3
20c8c03 section 6: review file
a47a3c3 step 6.7: how much of each headline statistic one month is carrying
86281f1 step 6.6: the non-primary feature set's conditional statistics, bootstrap and grid
```

That log is as of the review-file commit. This status file is committed and
pushed on top of it, so on the remote the head is one commit later:
`06: status file`.

Section 7 was not started.
