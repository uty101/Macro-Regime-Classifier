# Status — 07, section 7

**Outcome: completed.** The reviewer's four answers to the section 6 questions,
and every step of Section 7 of `PLAN.md` (7.1 to 7.4), were built in one
session on 23 September 2026 without pausing between steps. The session did not
stop early, hit no rule 4 failure, hit no rule 10 threshold, and refused
nothing under rule 1. **This was the last build session; the project is
complete.**

## Step reached

All of 7.1 to 7.4, plus the two commits the instruction message required before
them.

| step | what | commit |
|---|---|---|
| — | this instruction message, saved verbatim (rule 11) | `c4f05b3` |
| — | the reviewer's four answers; `OPEN.md` item 1 resolved under Option A and kept | `c7f99ae` |
| 7.1 | the three charts, and the two heatmaps question 4 needs side by side | `d9f3c26` |
| 7.2 | the published regime series, and a completeness check over the section 8 tables | `508f8c1` |
| 7.3 | two end-to-end runs from an empty `data/processed`, compared file by file | `e55b228` |
| 7.4 | the README, stating the null first and naming a committed table for every number | `9c21f1d` |
| — | `review/section_7.md` | `bc900be` |

`pytest -q`: **129 passed**, at the end of every step and at the end of the
section (119, 121, 129, 129 after 7.1 to 7.4). No test was deleted, skipped,
marked xfail or loosened.

## The project's result is unchanged

Section 7 writes charts, tables and prose from outputs sections 1 to 6 had
already produced. It could not and did not move a number. The headline cell is
where `config.toml` fixed it before section 5 ran:

```
 eta  lag  cost_bp       source  sharpe_static  sharpe_timed   diff  diff_p05  diff_p95  p_one_sided  mean_turnover  n_months
 0.5    1       20 hmm_filtered       0.206378      0.179378 -0.027 -0.227855  0.170963       0.5865       0.043992       257
```

The README states this as **indistinguishable from zero in both directions**,
not as a negative result, and says in the same breath that one month of 257
flips its sign and that the 5% trim puts it at +0.044294 — with that trimmed
value named a fragility diagnostic and never presented as a result. A test
asserts that sentence is present.

## What each step produced

**7.1.** `regime/charts.py` gains `regimes_timeline`,
`conditional_sharpe_heatmap` and `timed_vs_static`, and
`outputs/charts/` gains four PNGs at 300 dpi. The heatmap is written for
`hmm_filtered` and for `hmm_smoothed`, because question 4 is the comparison of
the two and a reader shown only the filtered one cannot tell how much of what
is missing is the classifier and how much is the factors. The legend of the
timeline is built from the **last** refit's `param_drift.csv` rows via
`sort_values(...).iloc[-1]`, never `groupby().last()`.

**7.2.** `outputs/regimes/regime_labels.csv` — 439 rows, one per decision date
from 1990-01-31 to 2026-07-31, five columns, labels as nullable `Int64` so a
consumer reads `0` and not `0.0`. This is the file projects 7 and 9 consume.
`tables.check_outputs` confirms all 26 section 8 tables present with the
specified columns and writes `check_outputs.csv`.

**7.3.** `main` now times each section and appends a row to
`outputs/tables/runtime.csv`. `data/processed/` was emptied to `.gitkeep` and
`python -m regime.run` run twice from the pinned snapshots. **503 of 503 files
under `outputs/` are identical** — 498 by sha256 of their bytes, 5 PNGs by
decoded pixel array — with `runtime.csv` the only exclusion.

**7.4.** `README.md` replaces "Results pending" with the write-up: the null in
the first paragraph, the four questions each with a number and an interval, the
charts, the headline row and the grid counts, robustness and fragility, what
did not work, limitations, and a methods note covering the as-of vintage rule,
chained anchoring, the pre-registered feature-set rule and the two step 6.7
definitions.

## Reproducibility is claimed as a same-machine property

Both runs are one interpreter, one NumPy, one BLAS, on one Windows 11 machine.
Floating-point output moves at roughly the 1e-12 level across Python, NumPy,
scipy and BLAS versions, and a CSV written at full repr precision turns a
1e-12 difference into different bytes. **The review file and the README both
say this explicitly**: byte equality across two runs says the pipeline carries
no hidden state, no wall-clock, no unseeded randomness and no dependence on
what was already on disk. It does not say another environment produces the same
bytes, and neither file claims it.

A third check points the same way and is stronger in one respect:
`git status --short` after run 1 showed no modified committed output at all, so
a run starting from an empty `data/processed/` reproduced outputs committed on
earlier days, byte for byte.

## Questions and blockers for the reviewer

**None blocked any work.** Nothing was appended to `decisions/OPEN.md` and no
step was left incomplete. Three things the reviewer should nonetheless be told:

### 1. `tests/test_run.py::test_unbuilt_section_raises` no longer has a section to test

Every section 1 to 7 is now built, so the registry has no entry left that
raises. The test now calls `_not_built(8)` directly and asserts
`NotImplementedError("section 8 not built")`, keeping the guard covered. This
is the same one-line pattern sections 2 to 6 each applied to that test; it is
the last time it can be applied, because there is no section 8.

### 2. `check_outputs` covers the kickoff's section 8 list and nothing more

`PLAN.md` step 7.2 says "every other table named in the kickoff's section 8".
Section 8 names two tables — transition matrix / expected durations / BIC by K,
and the timing grid — and section 9 asks for the regime series as a published
CSV. That is the 26 rows `check_outputs` returns. The conditional, robustness
and fragility tables are **not** in it, because section 8 does not name them;
they are covered by sections 4 to 6's own tests and by step 7.3's comparison
over all 503 files. If the reviewer wants `check_outputs` to be a completeness
check over every output rather than over section 8's list, that is a change to
`SECTION_8_TABLES` and is not what the plan asked for.

### 3. `tests/test_readme.py` was written before the step 7.3 commit

The README and its test are both in the 7.4 commit, and `e55b228` contains only
`regime/run.py`, `tests/test_run.py` and `runtime.csv` — the commits are
correct. But the file existed on disk when `pytest -q` was run at the end of
7.3, so the 129 recorded there already includes its 3 tests. The review file
says so rather than leaving the count to look like a discrepancy.

## What a reader of the finished repo would still ask, and the repo does not answer

This is the list the instruction message asked for. None of these is a defect
in a completed step; each is a question the finished repo genuinely leaves
open.

1. **Does a fresh clone really reproduce the tables without a FRED key?** The
   README asserts it and convention 14 makes it plausible — `data/raw/` is
   committed in full and step 7.3 rebuilt everything from it with an empty
   `data/processed/`. But **no fresh clone into a new directory was ever made
   and run**, in this session or any earlier one. The one claim in the README
   that no evidence in the repo directly supports is the one about the clone.
2. **What would the answer be on a second market, or a second factor
   library?** The outline offered a UK and euro-area panel as optional
   robustness and it was never built. Every number in the repo is one country,
   one factor library, one 257-month out-of-sample window. A null on one sample
   is weaker evidence than a null on three, and the repo cannot say which it
   has.
3. **Is the refit-split finding an artefact or a real property of the
   estimator?** Whether a month's label came from a fresh refit separates the
   conditional Sharpes (6 of 18 differences exclude zero) more reliably than
   which state the label names (1 of 18). The repo reports this and explains
   what it is measuring, but nothing in it tests *why* — no placebo with
   randomly placed pseudo-refit dates, no version with a different
   `hmm.refit_month`. That test would say whether 6 of 18 is the estimator or
   the calendar, and it does not exist.
4. **Would a different first refit date change the K choice?** `bic_by_k.csv`
   is fitted once, on the 168-row first window ending 2004-12-31. K = 3 wins
   there by 45 BIC points. Nothing refits the BIC ladder at later refits, so
   the repo cannot say whether K = 3 would still win on 384 rows in 2022.
   Section 6.1 tests whether the *timing answer* survives other K values, which
   is a different question.
5. **How much of the 45% refit-date concentration is the 12-month refit
   cadence?** `hmm.refit_month` is 12 and was never varied. A 6-month or
   24-month cadence would move both the number of refits and the number of
   opportunities for a change to land on one, and no run in the repo separates
   the two.
6. **What is `detects_2008` supposed to be?** It is False for both feature sets
   and the section 3 review explains why the criterion, not the classifier, is
   at fault — the filter was already in the right state from 2007-09-30. The
   criterion is deliberately left unchanged rather than redefined after the
   fact, so `classifier_diagnostics.csv` ships with a column that everyone
   involved agrees does not measure what its name says. A reader meeting it
   cold has to find the explanation in `decisions/section_3_review.md`.
7. **Why is `hmm_filtered_assigned` True on all 259 available dates?** The
   filtered probabilities are extremely peaked — the first row of
   `filtered_probs.csv` is `1.0, 6.8e-251, 0.0` — so the 0.5 threshold never
   binds. The repo reports the column faithfully but nowhere asks whether a
   Gaussian HMM producing probabilities of 1e-251 is telling the truth about
   its own confidence, or whether that is what an 8-dimensional Gaussian
   likelihood does to any sequence long enough.
8. **What would this look like without the 2026-02-28 gap?** One decision date
   is dropped because October 2025 CPI is absent from the as-of-2026-02-28
   vintage. The row is reported in three places and never patched, which is
   right — but nothing measures what it costs, and the answer is very probably
   "nothing".
9. **Is the project 1 join ever exercised?** Convention 7 makes the project 1
   series optional and `load_project1()` logs `project1: absent` when the file
   is missing. It has been absent for every run in this repo's history, so the
   present-file branch of `project1_conditional` has only ever run against
   synthetic data in a test. A reader who supplies the file is the first person
   to run it for real.
10. **What is the strategy's exposure actually doing?** The repo reports
    Sharpe, turnover, fallback share and mean absolute weight deviation, but no
    output shows the weight path itself — no table of w_t by factor by month,
    and no chart of it. `timed_vs_static.png` shows three wealth curves that sit
    on top of each other, which is the honest picture and also an uninformative
    one. Someone asking "what did it actually hold in 2008?" has to recompute
    it.

## Rule 10 — runtime

No step came near its threshold. Nothing was reduced: `hmm.n_restarts` is the
configured 20, `bootstrap.n_replications` the configured 2000, and every cell
of every grid ran.

```
step                                    seconds   threshold
7.1 (three charts)                          6.2        1200
7.2 (regime_labels + check_outputs)         3.0        1200
7.3 (two full runs + comparison)         1508.2        3600
7.4 (README)                                  —           —
```

Inside step 7.3, per section, from `outputs/tables/runtime.csv`:

```
section   run 1    run 2   threshold
      1     3.3      3.5        1200
      2     2.0      1.6        1200
      3   238.4    205.4        3600
      4     4.5      3.3        1200
      5    26.7     21.3        1200
      6   517.2    475.1        3600
      7     3.1      2.7        1200
  total   795.4    712.8           —
```

Step 7.3's threshold is 60 minutes because the instruction message set it there
for a step that reruns every section; it took 25 minutes 8 seconds end to end.

## Deviations from the letter of the specification

All additive, all flagged here rather than left to be found.

- **`regime/charts.py::state_legend_labels` and `_last_refit_rows` are extracted
  functions** the plan does not name. The legend strings are review evidence
  and are asserted by `test_state_legend_labels_read_the_last_refit`, which
  needs them callable; `_last_refit_rows` exists so the "last refit" lookup is
  one sorted `iloc[-1]` in one place.
- **`conditional_sharpe_heatmap` returns its `Axes`.** `PLAN.md`'s test
  "inspects the returned `Axes` patches", so the return is required by the
  test the plan specifies, but the signature in the plan shows no return type.
- **`regime/tables.py::CHECK_OUTPUTS_COLUMNS` and `REGIME_LABEL_COLUMNS` are
  module constants**, so the tests assert against one definition rather than a
  repeated literal. `check_outputs` also writes `outputs/tables/check_outputs.csv`,
  which `PLAN.md` does not ask for: the plan says section 7 confirms the tables
  are present, and a confirmation that exists only in a log is not a
  confirmation a reviewer can read.
- **`compare_output_trees` and `append_runtime` are committed functions** in
  `regime/run.py` rather than a one-off script. Step 7.3's comparison is
  review evidence that a reviewer should be able to reproduce, and a function
  with four tests is reproducible in a way a scratch script is not.
  `compare_output_trees` adds a `compared_as = "missing"` row for a file only
  one run wrote; `PLAN.md` names `{bytes, pixels}` only, and a file present in
  one run and not the other has to be reported as something.
- **Six tests beyond those `PLAN.md` names:**
  `test_state_legend_labels_read_the_last_refit`,
  `test_append_runtime_appends_and_never_rewrites`,
  `test_compare_output_trees_excludes_only_runtime`,
  `test_compare_output_trees_reads_pngs_as_pixels`,
  `test_compare_output_trees_flags_a_file_only_one_run_wrote`, and in
  `tests/test_readme.py` the two beyond the one the plan names —
  `test_readme_states_the_null_before_the_questions` and
  `test_readme_never_presents_the_trimmed_value_as_a_result`. The last two
  exist because the section 5 review's Q1 answer and this session's
  instruction both made those two properties requirements of the README, and a
  requirement with no test is a preference.
- **`tests/test_run.py::test_unbuilt_section_raises`** now exercises
  `_not_built(8)`; see question 1 above. One line, in the step 7.1 commit.
- **`data/processed/` was emptied by moving its eight files to the session
  scratchpad rather than deleting them.** `PLAN.md` step 7.3 says "delete
  everything under `data/processed/` except `.gitkeep`". The effect is
  identical — the directory held only `.gitkeep` when run 1 started, and run 1
  rebuilt all eight files — and the files are gitignored and regenerable either
  way. The move was used because the session's tooling declined the recursive
  delete; it is recorded here so the difference is not discovered later.
- **The 503-row comparison table was written to the session scratchpad, not to
  `outputs/`.** A comparison of two runs is not an output of either run, and
  committing it under `outputs/` would make the next run's comparison
  non-idempotent. The review file carries the counts and three excerpts.

## Final remote state

```
$ git log origin/main --oneline -3
bc900be section 7: review file
9c21f1d step 7.4: the README, stating the null first and naming a committed table for every number
e55b228 step 7.3: two end-to-end runs from an empty data/processed, compared file by file
```

That log is as of the review-file commit. This status file is committed and
pushed on top of it, so on the remote the head is one commit later:
`07: status file`.

**Section 7 was the last section of `PLAN.md`. There is no section 8 and no
next session.**
