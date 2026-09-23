Session 7 — build Section 7 of PLAN.md (steps 7.1 to 7.4), all steps, in this session. This is the last build session. Do not pause between steps. Complete everything unless rule 4 applies, then write the review file and the status file and push.

Per rule 11: first save this message verbatim as instructions/07_section_7.md and commit it as `07: instructions`.

Then one commit `section 7: reviewer decisions`, text only, recording in decisions/section_6_review.md:
- OPEN.md item 1: Option A. K = 5 is reported as a variant that cannot be filtered; no change to hmm_numpy.py. The reason: 240 free parameters on 192 training rows collapses a state onto a 7-dimensional hyperplane, and forward_filter refusing a singular covariance is correct behaviour. The failure is the finding. Mark item 1 resolved in decisions/OPEN.md with the choice and this reasoning; do not delete it.
- Q2: both step 6.7 definitions accepted as implemented (floor on the trim count; pairwise influence measured by how far a month's removal moves the statistic, since the 2 states occupy disjoint months). Both are stated in the README's methods note.
- Q3: the covars_for_setter fix is accepted, and flagging a change to main-run code was the right call.

Then build steps 7.1 to 7.4 exactly as PLAN.md specifies. One commit per step, `step 7.Y: <one line>`. Section 7 of run.py runs 7.1 to 7.3.

Before starting: read CLAUDE.md, PLAN.md (the conventions block and Section 7 in full), config.toml, docs/CONVENTIONS_RESOLVED.md, and the 4 decisions/section_N_review.md files.

Session-specific points:
- Step 7.3's idempotence check runs twice on this machine in this session. Floating-point output differs across Python and BLAS versions at the 1e-12 level, so byte equality is a same-machine property; say so in the review file rather than claiming reproducibility across environments.
- Step 7.4's README, in order: (1) an opening paragraph stating the null — the filtered classifier's labels move at refits as often as not, it separates factor premia no better than chance, and no cell of the timing grid is distinguishable from zero; (2) what was built and how to run it; (3) the 4 questions, each with a number and an interval; (4) the 3 charts; (5) the headline row and the grid counts; (6) robustness_summary.csv and fragility_summary.csv; (7) what did not work; (8) limitations; (9) a short methods note covering the as-of vintage rule, chained anchoring, the pre-registered feature-set rule, and the 2 step 6.7 definitions.
- Question 4's answer is the filtered-against-smoothed comparison: the gap table, the 1 against 3 pairwise differences, and the smoothed labelling's timing cells, which also fail to clear zero. Hindsight helps the classifier and still does not produce a tradeable gain; say both halves.
- The timing result is stated as indistinguishable from zero in both directions, not as negative: the point estimate is −0.027, one month of 257 flips its sign, and the 5% trim puts it at +0.044. The trimmed value is never presented as a result.
- No number in the README may appear that is not in a committed table, and every number in it names the table it comes from.
- The README states that the repo's raw data snapshots are pinned and committed, so a fresh clone reproduces every table without a FRED key, and gives the 3 attribution lines already in the file.
- Do not publish the README as a hosted artifact or anything outside the repo.
- Any "latest" or "last" lookup uses iloc[-1], tail(1) or nth(-1), never groupby().last() or .first().
- pytest -q must pass at the end of every step. Threshold 20 minutes per step, 60 for step 7.3 if it reruns every section.

Write review/section_7.md from review/TEMPLATE.md; every number has its raw rows. Then write instructions/07_section_7.status.md per rule 11, and in it list anything a reader of the finished repo would still ask that the repo does not answer. Commit, push, print git log origin/main --oneline -3, and stop.
