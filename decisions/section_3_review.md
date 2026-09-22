# Decision — the reviewer's answers to the section 3 questions

The four questions are those in `instructions/03b_section_3_revision.status.md`,
"Questions and blockers for the reviewer". The reviewer answered them in the
session 4 instruction message (`instructions/04_section_4.md`); this file
records the answers so nothing in section 4 onward has to go back to chat for
them. None of these answers changes a convention, a config value or an output
already produced.

## Q1 — 9 of 20 filtered label changes land on a refit date (45%)

**Accepted as a finding, not an artefact to engineer away.**

Chaining removed the renaming component: the sorted run had 15 of 26 changes on
refit dates (58%), the chained run has 9 of 20 (45%). What remains is
re-estimation, not relabelling. It is concentrated in state 0 against state 2,
which are separated mainly by `dgs10_level`, whose boundary moves at each
December refit.

Against a uniform base rate of 1 in 12 (8.3%), 45% is far above chance. That
number is reported in the write-up as a finding, and its **cost is measured in
step 4.2** through `conditional_stats_refit_split.csv` (amendment 3 below): the
out-of-sample dates are split by whether the filtered run containing them began
on a refit date, and the conditional statistics are reported for each half. No
further pass over section 3 is made.

## Q2 — `detects_2008` is false for both feature sets

**The criterion was the reviewer's error, not a fault in the classifier.**

Under the 2007-12-31 refit, state 0 is low level, steep slope, weak `indpro`,
weak dollar, elevated `vix`. The filter entered state 0 at **2007-09-30**, the
month of the first Fed cut, and held it through 2008. The Q4 2008 extremes are
in the same direction as state 0, so with K = 3 there is no further state for
the filter to move into: the criterion asked for a change of state between
2008-06-30 and 2008-11-30 that a correctly-behaving K = 3 filter should not
make, because it was already in the right state and had been since 2007-09.

`detects_2008` therefore stays in `classifier_diagnostics.csv` **unchanged**,
with this note. It is not redefined after the fact, and the pre-registered rule
that consumed it is not re-run.

## Q3 — should the d = 7 diagnostics the rule does not weigh count?

**No. The primary feature set stays `"core"` (d = 8).**

d = 7 has fewer refit-date changes, but it misses 2020, and its filtered and
smoothed labels agree on only **40%** of dates against **68%** for d = 8. More
decisively: the pre-registered rule is not rewritten after its outcome is
known. That is the whole point of pre-registering it (convention 15).

d = 7 remains the step 6.6 rerun. `decisions/primary_feature_set.md` stands as
written.

## Q4 — the three deviations from the letter of the specification

**All three accepted.** The three extra tests, the `chain_to` parameter on
`run_smoothed_hmm` and `run_expanding_gmm`, and `median_run_months` being
computed over the whole filtered series are all additive, all flagged, and all
already reflected in `PLAN.md`.

## What this file does not do

It does not reopen `decisions/primary_feature_set.md`, does not change
`config.toml`, and does not cause any section 1 to 3 output to be regenerated
with different content. Sections 1 to 3 were re-run at the start of session 4
only to rebuild `data/processed/` and `outputs/`, which are not committed
(convention 14).
