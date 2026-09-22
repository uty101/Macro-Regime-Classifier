Session 4 — build Section 4 of PLAN.md (steps 4.1 to 4.5), all steps, in this session. Do not pause for confirmation or review between steps. Complete every step without stopping unless rule 4 applies, then write the review file and the status file and push.

Per rule 11: first save this message verbatim as instructions/04_section_4.md and commit it as `04: instructions`.

Then, before step 4.1, one commit `section 4: reviewer decisions and plan amendments`, text only:

1. decisions/section_3_review.md, recording the reviewer's answers to the 4 questions in instructions/03b_section_3_revision.status.md:
   - Q1 (9 of 20 filtered changes on refit dates): accepted as a finding, not an artefact to engineer away. Chaining removed the renaming (15 of 26 before); the remainder is re-estimation, concentrated in state 0 against state 2, which are separated mainly by dgs10_level, whose boundary moves at each December refit. Against a uniform base rate of 1 in 12, 45% is reported in the write-up and its cost is measured in step 4.2.
   - Q2 (detects_2008 false for both sets): the criterion was the reviewer's error. Under the 2007-12-31 refit, state 0 is low level, steep slope, weak indpro, weak dollar, elevated vix; the filter entered it 2007-09-30 (the first Fed cut) and held it through 2008. The Q4 2008 extremes are the same direction as state 0 with K = 3. The criterion stays in classifier_diagnostics.csv unchanged, with this note.
   - Q3: primary stays "core". d = 7 has fewer refit-date changes but misses 2020 and its filtered and smoothed labels agree on 40% of dates against 68% for d = 8. The pre-registered rule is not rewritten after its outcome is known. d = 7 remains step 6.6.
   - Q4: the 3 deviations are accepted.
2. PLAN.md step 4.3 also writes outputs/tables/conditional_differences_<source>.csv with columns factor, state_a, state_b, sharpe_diff, diff_p05, diff_p95, excludes_zero, n_a, n_b: for every factor and every pair of states a < b, sharpe_b − sharpe_a, with percentiles from the same bootstrap replications as the conditional Sharpes. It also writes outputs/tables/unconditional_stats.csv: factor, n, ann_mean, ann_std, sharpe, sharpe_p05, sharpe_p95 over the same out-of-sample dates, all months included.
3. PLAN.md step 4.2 also writes outputs/tables/conditional_stats_refit_split.csv for hmm_filtered only: every out-of-sample date is flagged run_started_on_refit (true if the filtered run containing it began on a refit date); conditional stats by factor, state and flag, with n. Information only, no bootstrap.
4. PLAN.md step 7.4 answers question 2 with the count of pairwise differences whose interval excludes zero, for filtered and for smoothed, not the per-cell count, and reports the refit-split table beside it.
5. tests/test_conditional.py::test_pairwise_difference_zero_when_states_identical: 2 states with identical return rows give sharpe_diff 0 and both percentiles 0. tests/test_conditional.py::test_run_started_on_refit_flag on a hand-built 8-row label series with 2 refit dates.

Then build steps 4.1 to 4.5 exactly as PLAN.md specifies. One commit per step, `step 4.Y: <one line>`. Section 4 of run.py runs 4.1 to 4.5.

Before starting the steps: read CLAUDE.md, PLAN.md (the conventions block and Section 4 in full), config.toml, docs/CONVENTIONS_RESOLVED.md, decisions/primary_feature_set.md and decisions/section_3_review.md. Rebuild data/processed and outputs with python -m regime.run --section 1, 2 and 3 (no --pull). All label sources come from the primary feature set's main outputs.

Session-specific points:
- Out-of-sample window: decision dates from 2004-12-31 to the last decision date with a t+1 factor return (2026-06-30 with sample.end 2026-07-31). Every source uses the same dates.
- In a bootstrap replication where a state has fewer than 2 observations, its Sharpe is NaN for that replication. Percentiles use np.nanquantile. Report per (source, factor, state) the number of NaN replications; if any exceeds 5% of replications, list it in the status file.
- StationaryBootstrap receives seed=cfg.run_seed and block_size=cfg.bootstrap_block_size. The filtered and smoothed labels are resampled in the same draw for step 4.4.
- Smoothed and filtered state numbers already agree through chained anchoring. Do not relabel anything.
- The 4 sources are hmm_filtered, hmm_smoothed, gmm_filtered and rules. Rules has 4 states and 6 pairs; HMM and GMM have primary_K states and 3 pairs.
- In addition to the plan's evidence: unconditional_stats in full; per source a factor × state pivot of Sharpe with n in brackets; for hmm_filtered and hmm_smoothed the conditional_differences tables in full; conditional_stats_refit_split in full; the mean monthly return of each factor in each state next to its unconditional mean, and the 3 largest absolute monthly returns inside each state with their dates, to show whether any state difference rests on a handful of months.
- Any "latest" or "last" lookup uses iloc[-1], tail(1) or nth(-1), never groupby().last() or .first().
- pytest -q must pass at the end of every step. The threshold for every Section 4 step is 20 minutes.

Write review/section_4.md from review/TEMPLATE.md; every number has its raw rows. Then write instructions/04_section_4.status.md per rule 11, commit, push, print git log origin/main --oneline -3, and stop. Do not start Section 5.
