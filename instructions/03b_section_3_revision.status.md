# Status — 03b, section 3 revision

**Outcome: completed.** Every item of `03b_section_3_revision.md` was built.
The session did not stop early, did not refuse anything under rule 1, and hit
no rule 4 or rule 10 threshold.

## Step reached

All of items 0 to 4. In order:

| item | what | commit |
|---|---|---|
| 0 | `instructions/`, rule 11 in `CLAUDE.md`, `00_history.md` back-fill | `748d558` |
| 1 | chained anchoring; `anchor_chain.csv`; GMM and smoothed chained; tests | `db7b686` |
| 2 | `features.primary`; both feature-set runs; `classifier_diagnostics.csv`; the rule applied; `decisions/primary_feature_set.md`; PLAN and conventions edits | `db7b686` |
| 3 | Revision section of `review/section_3.md` | `db7b686` |
| 4 | `pytest -q` green, revision committed, this file | `db7b686` + this commit |

`pytest -q`: **65 passed**. No factor return was loaded, read or joined at any
point in this session.

## Result of the pre-registered rule

`features.primary` stays **`"core"`** (d = 8). The d = 7 row fails
`detects_2008` and `detects_2020`, so the third condition never applied.
`config.toml` already held that value, so nothing was regenerated and the
d = 8 run remains the main output.

```
feature_set,n_filtered_changes,n_changes_on_refit_dates,share_on_refit_dates,median_run_months,max_expected_duration,n_infinite_durations,n_degenerate_states,max_matched_distance,detects_2008,detects_2020,filtered_smoothed_agreement
core,20,9,0.45,10.0,129.87537037913293,1,0,2.6893908073181687,False,True,0.6821705426356589
core_no_level,21,7,0.3333333333333333,9.5,57.10272796063988,0,0,2.1314511105461564,False,False,0.40310077519379844
```

## Questions and blockers for the reviewer

None blocked the work. Four things need a decision before section 4, in
descending order of importance:

1. **9 of the primary set's 20 filtered label changes still land on a refit
   date, and every one of them is a 31 December.** Chaining cut this from 15 of
   26, but a regime that begins on the day the model was refitted and on no
   other day of the year is still partly an artefact of the refit schedule.
   Section 4's conditional statistics will inherit it. Is that acceptable, or
   does it want another pass first?
2. **`detects_2008` is false for both feature sets.** Neither filter changes
   state between 2008-06-30 and 2008-11-30; both move at the 2008-12-31 refit.
   The decision rule's first condition is therefore one that neither candidate
   met, which weakens the rule as a discriminator even though it was applied
   exactly as written.
3. **The d = 7 run is better on three diagnostics the rule does not weigh**:
   fewer refit-date changes (7 vs 9), a lower share (33% vs 45%), no infinite
   durations and no degenerate states. The rule was fixed in advance and was
   applied mechanically, so this session did not act on any of that. If the
   reviewer thinks those columns should count, that is a change to
   `decisions/primary_feature_set.md` and convention 15, not a session choice.
4. **Three deviations from the letter of the specification**, all additive, all
   flagged here rather than left to be discovered:
   - Three tests were added that the specification did not list:
     `test_chaining_survives_a_feature_the_sort_rule_cannot_separate`,
     `test_primary_feature_set_rule_needs_all_three_conditions` and
     `test_label_runs_lengths_and_starts`. A pre-registered rule and the
     revision's central claim should each have a test.
   - `run_smoothed_hmm` and `run_expanding_gmm` gained a `chain_to` parameter.
     The specification requires the chaining but did not restate those
     signatures; PLAN.md steps 3.5 and 3.6 were updated to match.
   - `median_run_months` is computed over the whole filtered series, while the
     change counts start at 2005-01-31 as specified. The first out-of-sample
     date has no predecessor and so cannot be a change, but it does belong to a
     run. Stated in the docstring and in the review.

## Final remote state

```
$ git log origin/main --oneline -3
db7b686 section 3: chained anchoring and pre-registered primary feature set
748d558 workflow: instructions and status live in the repo
13730b1 Add the project outline document
```

That log is as of the revision commit. This status file is committed and
pushed on top of it, so on the remote the head is one commit later:
`03b: status file`.
