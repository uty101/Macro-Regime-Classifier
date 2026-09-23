Session 6 — build Section 6 of PLAN.md (steps 6.1 to 6.6), plus a new step 6.7, all steps, in this session. Do not pause between steps. Complete everything unless rule 4 applies, then write the review file and the status file and push.

Per rule 11: first save this message verbatim as instructions/06_section_6.md and commit it as `06: instructions`.

Then one commit `section 6: reviewer decisions and step 6.7`, text only, recording the reviewer's answers to the 3 questions in instructions/05_section_5.status.md in decisions/section_5_review.md:
- Q1: yes. Step 7.4's README states the null in its first paragraph, before the 4 questions are worked through: the filtered classifier's labels move mostly at refits, it separates factor premia no better than chance, and no cell of the timing grid is distinguishable from zero. PLAN.md step 7.4 amended to say so.
- Q2: accepted as a property of the design, with a reporting requirement. PLAN.md step 6.4 also reports, per min_regime_obs setting, the number and share of out-of-sample months in fallback and the mean absolute weight deviation from 1/5, beside the diff. A larger diff at 12 is read as more trading, not more signal, unless the fallback share says otherwise.
- Q3: yes, and it becomes step 6.7 below.

New PLAN.md step 6.7 — fragility of the headline statistics.
  Builds: regime/robustness.py::leave_one_month_out(timed: pd.Series, static: pd.Series, cfg) -> pd.DataFrame — for each of the 257 earning months of the headline cell (eta 0.5, lag 1, 20 bp, hmm_filtered), recompute diff = sharpe(timed) − sharpe(static) with that month dropped from both series; columns dropped_month, diff, sign_flipped. And trimmed_diff(timed, static, trim=0.05) -> float — both Sharpes recomputed after removing the 5% of months with the largest absolute timed-minus-static difference, dropped from both series so they stay aligned.
  Also applies the same 2 statistics to the one section 4 pairwise difference that excludes zero (hmm_filtered, UMD, state 0 against state 2): leave-one-month-out over the months of both states, and the 5% trimmed version, via regime/robustness.py::leave_one_month_out_pairwise(labels, factors, factor, state_a, state_b, cfg).
  Outputs: outputs/tables/robustness/leave_one_month_out_headline.csv, leave_one_month_out_umd_0_2.csv, and outputs/tables/robustness/fragility_summary.csv with one row per statistic: statistic, full_value, loo_min, loo_max, n_sign_flips, share_sign_flips, trimmed_value, n_months.
  Tests: tests/test_robustness.py::test_loo_returns_one_row_per_month, ::test_loo_full_value_matches_the_grid (the un-dropped statistic equals timing_results.csv's diff to 1e-12), ::test_trimmed_diff_drops_the_right_count_from_both_series.
  Review evidence: fragility_summary.csv in full; the 10 months whose removal moves the headline diff most, with their timed and static net returns; the same for the UMD pair.
  6.7 is added to run.long_steps only if it exceeds 20 minutes; measure first.

Then build steps 6.1 to 6.7 exactly as PLAN.md specifies. One commit per step, `step 6.Y: <one line>`. Section 6 of run.py runs 6.1 to 6.7.

Before starting: read CLAUDE.md, PLAN.md (the conventions block and Section 6 in full), config.toml, docs/CONVENTIONS_RESOLVED.md, and the 3 decisions/section_N_review.md files.

Session-specific points:
- Every robustness run is reported against the headline cell of the main run. None of them can replace it. No sentence in the status or review file may present a robustness variant as the project's result, whatever it shows.
- Step 6.1 (K in 2, 3, 4, 5) and step 6.6 (the non-primary feature set) rerun the full classifier, conditional statistics and the timing grid. Step 6.2 (diag) and step 6.3 (the 10-feature run from 2009-12-31, diag) likewise. Each writes under outputs/tables/robustness/<name>/ and outputs/regimes/robustness/<name>/, and each contributes its headline-equivalent row to outputs/tables/robustness/robustness_summary.csv: variant, sharpe_static, sharpe_timed, diff, diff_p05, diff_p95, p_one_sided, mean_turnover, fallback_share, n_months, plus n_pairwise_excl_zero and n_changes_on_refit_dates so the classifier's behaviour travels with its timing number.
- If any variant produces a headline-equivalent diff with p_one_sided below 0.1, say so plainly and then check, before writing anything else about it, that its weights use only s < t, that its lag is applied as specified, and that its static comparator runs on identical earning months. Report the check in the status file. A single variant clearing 0.1 out of roughly 10 is what chance produces.
- Step 6.5 sweeps bootstrap block_size over 3, 6 and 12 for the headline cell only, as planned.
- Any "latest" or "last" lookup uses iloc[-1], tail(1) or nth(-1), never groupby().last() or .first().
- pytest -q must pass at the end of every step. Threshold 20 minutes per step, 60 for 6.1, 6.2, 6.3 and 6.6.

Write review/section_6.md from review/TEMPLATE.md; every number has its raw rows. Then write instructions/06_section_6.status.md per rule 11, commit, push, print git log origin/main --oneline -3, and stop. Do not start Section 7.
