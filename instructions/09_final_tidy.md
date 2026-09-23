Session 9 — 1 addendum and the final tidy-up. Build it in this session, then write the status file and push. Rule 4 applies.

Per rule 11: first save this message verbatim as instructions/09_final_tidy.md and commit it as `09: instructions`.

1. decisions/OPEN.md item 1 addendum, under the existing resolution, not replacing it: K = 5 surfaces its rank deficiency as 1 of 2 exceptions depending on the library stack — numpy.linalg.LinAlgError from scipy's positive-definite check on one, ValueError("component 4 of 'full' covars must be symmetric, positive-definite") from hmmlearn's _utils._validate_covars on another. Both are recorded as status = singular_covariance by is_singular_covariance (step 8.2). The resolution is unchanged: Option A, report the failure, no pseudo-inverse, nothing in hmm_numpy.py moves.

2. README, reproducibility paragraph: record that a clean clone with FRED_API_KEY unset was run independently by the reviewer on a different Python and BLAS build, that pytest gave 132 passed, that python -m regime.run completed end to end and wrote all 5 charts and every table, that the K = 5 variant was recorded rather than fatal there, and that the headline row and fragility_summary.csv matched the committed values at every printed digit. Keep it to a short paragraph and keep every existing claim that is still true.

3. docs/CONVENTIONS_RESOLVED.md, add item 17: the repo is reproducible from a clean clone with no FRED key because data/raw/ is committed and config.toml pins each pull_id; tests build what they read rather than assuming data/processed/ is populated; byte-identical output is a same-machine property, while the reported results match across environments at every printed digit.

4. Confirm no loose ends and report each in the status file: decisions/OPEN.md has no unresolved item; every instructions/NN_*.md has its matching .status.md; git status is clean after python -m regime.run; README's reproducibility paragraph is the only place claiming an external verification and it now names one that happened.

pytest -q must pass. Threshold 20 minutes. Write instructions/09_final_tidy.status.md per rule 11, and in it state plainly whether the project is complete and what remains, if anything. Commit, push, print git log origin/main --oneline -3, and stop.
