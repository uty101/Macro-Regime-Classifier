# Status — 04, section 4

**Outcome: completed.** Every step of Section 4 of `PLAN.md` (4.1 to 4.5) was
built, in one session, without pausing between steps. The session did not stop
early, did not refuse anything under rule 1, and hit no rule 4 or rule 10
threshold.

## Step reached

All of 4.1 to 4.5, plus the two text-only commits the instruction message
required before them.

| step | what | commit |
|---|---|---|
| — | this instruction message, saved verbatim (rule 11) | `13dfcfe` |
| — | reviewer decisions and the PLAN amendments they imply | `f6cd731` |
| 4.1 | `join_next_return`, the t+1 join and the monthly-index assertion | `c3ed4d1` |
| 4.2 | `conditional_stats` for the four sources; `conditional_stats_refit_split.csv`; section 4 of `run.py` | `14cba40` |
| 4.3 | `bootstrap_conditional`; the nine-column tables; `conditional_differences_<source>.csv`; `unconditional_stats.csv`; `bootstrap_nan_replications.csv` | `034b24a` |
| 4.4 | `filtered_smoothed_gap` | `895a41c` |
| 4.5 | `project1_conditional`, absent branch and present branch | `4a82ffb` |
| — | `review/section_4.md` | `5b04db6` |

`pytest -q`: **81 passed**, at the end of every step and at the end of the
section.

Sections 1, 2 and 3 were re-run first with `python -m regime.run --section N`
and no `--pull`. The `classifier_diagnostics.csv` they printed is identical to
the one recorded in `instructions/03b_section_3_revision.status.md`, so section
4 sits on the same classifier the reviewer cleared.

## What the section found

The out-of-sample window is **258 decision dates, 2004-12-31 to 2026-06-30**,
identical for all four label sources, with **no unassigned date** in any of
them.

The count that answers PLAN.md question 2, as amended (pairwise differences,
not per-cell Sharpes):

```
      source  cells_excl_zero  cells  diffs_excl_zero  diffs
hmm_filtered                4     18                1     18
hmm_smoothed                4     18                3     18
gmm_filtered                3     18                3     18
       rules                5     24                7     36
```

`hmm_filtered` is the only source the strategy may use out of sample, and it
separates the factor premia least: **1 of 18** pairwise differences excludes
zero (UMD, state 0 against state 2, −1.024, [−1.740, −0.112]). At a 90%
interval roughly 2 of 18 would be expected by chance, so the filtered HMM finds
fewer regime-conditional differences than noise would produce. The hindsight
smoothed labelling finds 3, and the backward-looking rules quadrants 7 of 36.

The one difference `hmm_filtered` does find leans on a single month: state 2's
UMD total is −0.419 and its largest month is 2009-03-31 at −0.3436, the
momentum crash. Removing the three largest months turns that state's mean from
−0.40 bp to +17 bp a month. This is in the review file, not smoothed over.

## Rule 10 — runtime

No step came close to its 20-minute threshold. Nothing was reduced:
`bootstrap.n_replications` is the configured 2000 and `bootstrap.block_size`
the configured 6 in every table.

```
step  seconds
 4.1 0.023559
 4.2 0.079947
 4.3 2.360327
 4.4 0.593575
 4.5 0.002165
```

## NaN replications (the session's 5% reporting threshold)

Per (source, factor, state), the number of bootstrap replications in which the
state held fewer than two rows and its Sharpe was therefore NaN. **Nothing
exceeds 5%; nothing exceeds 0.05%.** The only affected cells:

```
      source factor  state  n_replications  n_nan  share_nan
gmm_filtered Mkt-RF      2            2000      1     0.0005
gmm_filtered    SMB      2            2000      1     0.0005
gmm_filtered    HML      2            2000      1     0.0005
gmm_filtered    RMW      2            2000      1     0.0005
gmm_filtered    CMA      2            2000      1     0.0005
gmm_filtered    UMD      2            2000      1     0.0005
```

`gmm_filtered` state 2 holds 19 of the 258 dates, so one draw in 2000 contained
fewer than two of them. The full table is
`outputs/tables/bootstrap_nan_replications.csv`.

## Questions and blockers for the reviewer

Nothing blocked the work and nothing was appended to `decisions/OPEN.md`. Four
things want a decision before section 5, in descending order of importance:

1. **Section 5 builds a timed strategy on a classifier that does not separate
   the factor premia.** `hmm_filtered` produces 1 of 18 pairwise state
   differences whose interval excludes zero, below the ~2 of 18 a 90% interval
   would give by chance, and that one depends on the 2009-03 momentum crash
   sitting inside state 2. Section 5 will still run — the plan fixes it and
   rule 1 forbids a session deciding otherwise — but the reviewer should know
   before it does that the headline timing result is most likely to be a null,
   and may want the README framed around that rather than around a number that
   turns out positive by construction of the grid.

2. **The refit split is large and the reviewer may want it bootstrapped.** Per
   the amendment it carries no interval, by specification. But RMW state 0
   splits 3.228 (n = 22, runs that did not begin on a refit) against 0.263
   (n = 69, runs that did), and state 0's pooled 0.944 — one of the four cells
   that excludes zero — is an average of those. HML state 2 splits 1.313
   against −0.463 and CMA state 1 −0.549 against 2.016. If the reviewer wants
   to claim anything from this table rather than merely report instability, it
   needs replications; that is a PLAN change, not a session choice.

3. **`unconditional_stats.csv` makes several conditional cells look weaker than
   they read.** Mkt-RF pays 0.668 unconditionally over this window with an
   interval excluding zero, so `hmm_filtered` Mkt-RF state 2 at 0.712 is
   essentially the pooled number. Of the four `hmm_filtered` cells that exclude
   zero, two are Mkt-RF. The reviewer may want step 7.4 to report the
   conditional Sharpe minus the unconditional one rather than the level; that
   would be a further amendment.

4. **One deviation from the letter of the specification**, additive, flagged
   here rather than left to be found:
   - Seven tests beyond those the plan and the instruction message list:
     `test_join_drops_dates_before_first_window_end`,
     `test_join_rejects_a_gap_in_the_factor_index`,
     `test_pair_count_is_three_at_k3_and_six_at_k4`,
     `test_unconditional_stats_uses_every_month_and_the_same_dates`,
     `test_gap_sign_is_smoothed_minus_filtered`,
     `test_conditional_join_runs_when_present`, and the
     `unassigned_dates` assertion inside
     `test_unassigned_rows_are_excluded_and_counted`. The gap-sign test and the
     present-branch project 1 test exist because both would otherwise be code
     paths that had never executed.
   - `tests/test_run.py::test_unbuilt_section_raises` asserted that section 4
     was not built; it now asserts it of section 5. One line, in the step 4.2
     commit.
   - `bootstrap_conditional` returns three frames rather than one, so the
     Sharpes, the pairwise differences and the NaN counts come from a single
     bootstrap. `block_bootstrap_ci` keeps the signature and return type
     `PLAN.md` specifies and is a thin wrapper over it.

## Final remote state

```
$ git log origin/main --oneline -3
5b04db6 section 4: review file
4a82ffb step 4.5: the project 1 conditional join, or a logged skip
895a41c step 4.4: the filtered-minus-smoothed Sharpe gap, both labellings in one draw
```

That log is as of the review-file commit. This status file is committed and
pushed on top of it, so on the remote the head is one commit later:
`04: status file`.
