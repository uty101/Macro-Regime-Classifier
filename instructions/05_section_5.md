Session 5 — a section 4 addendum, then Section 5 of PLAN.md (steps 5.1 to 5.5), all steps, in this session. Do not pause between steps. Complete everything unless rule 4 applies, then write the review file and the status file and push.

Per rule 11: first save this message verbatim as instructions/05_section_5.md and commit it as `05: instructions`.

Then the addendum, one commit `section 4 addendum: refit-split intervals and excess Sharpe`, answering the 4 questions in instructions/04_section_4.status.md. Record the reviewer's answers in decisions/section_4_review.md: Q1 section 5 proceeds and the README is framed around the null found in section 4, not around whichever grid cell comes out positive; Q2 yes, the refit split gets intervals; Q3 yes, report excess over unconditional; Q4 the deviations are accepted.
   - conditional_stats_refit_split.csv (hmm_filtered only) gains sharpe_p05, sharpe_p95, excludes_zero per (factor, state, run_started_on_refit) and, per (factor, state), the difference between the two flags with its own p05, p95 and excludes_zero, written to outputs/tables/conditional_refit_split_differences.csv. Same StationaryBootstrap settings and seed as step 4.3, both flags resampled in one draw.
   - Every conditional_stats_<source>.csv gains excess_sharpe = sharpe − the factor's unconditional Sharpe from unconditional_stats.csv, and every conditional_differences_<source>.csv keeps its columns unchanged (a difference of 2 conditional Sharpes already nets the unconditional level out).
   - PLAN.md steps 4.2 and 4.3 updated to match; step 7.4 reports excess_sharpe beside sharpe and the refit-split differences beside the pairwise ones.
   - Tests: tests/test_conditional.py::test_excess_sharpe_is_conditional_minus_unconditional and ::test_refit_split_difference_zero_when_flags_identical.
   - Re-run sections 1 to 4 (python -m regime.run --section N, no --pull) and commit the regenerated tables with the addendum.

Then build steps 5.1 to 5.5 exactly as PLAN.md specifies. One commit per step, `step 5.Y: <one line>`. Section 5 of run.py runs 5.1 to 5.5.

Before starting the steps: read CLAUDE.md, PLAN.md (the conventions block and Section 5 in full), config.toml, docs/CONVENTIONS_RESOLVED.md, decisions/section_3_review.md and decisions/section_4_review.md.

Session-specific points:
- The headline cell is fixed in advance and is the only one the status file may lead with: η 0.5, lag 1, 20 bp, hmm_filtered. Report the other 71 rows as a grid, never as a result. No sentence in the status or review file may present the best cell of the grid as the finding.
- Section 4 found 1 of 18 pairwise differences excluding zero for hmm_filtered, below chance. A timing gain near zero, or negative after costs, is the expected outcome and is reported as such. If the headline diff is positive with p_one_sided below 0.1, do not treat that as confirmation: check first that the weights at t use only s < t (step 5.1's test), that the lag is applied as specified, and that the static comparator runs on identical earning months, and say in the status file that you checked.
- Add tests/test_strategy.py::test_timed_equals_static_when_all_sharpes_are_equal: when every trailing conditional Sharpe is identical across factors, timed weights equal 0.2 for every η and the net series matches static to 1e-12.
- In addition to the plan's evidence: per source and η, the number of months where the weight vector differs from static by more than 1e-9, and the mean and max absolute deviation of a weight from 0.2; the 10 largest single-month turnover values in the headline cell with their dates; sharpe_timed − sharpe_static for lag 0 and 0 bp beside the headline, so the cost of the lag and the cost of trading are separated.
- Any "latest" or "last" lookup uses iloc[-1], tail(1) or nth(-1), never groupby().last() or .first().
- pytest -q must pass at the end of every step. Threshold 20 minutes per step, except step 5.5 (72 cells × 2000 replications) which has 60.

Write review/section_5.md from review/TEMPLATE.md; every number has its raw rows. Then write instructions/05_section_5.status.md per rule 11, commit, push, print git log origin/main --oneline -3, and stop. Do not start Section 6.
