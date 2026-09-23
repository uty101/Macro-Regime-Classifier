# Decision — the reviewer's answers to the section 5 questions

The three questions are those in `instructions/05_section_5.status.md`,
"Questions and blockers for the reviewer". The reviewer answered them in the
session 6 instruction message (`instructions/06_section_6.md`); this file
records the answers so nothing in section 6 onward has to go back to chat for
them.

Two of the three change `PLAN.md`. Neither changes a convention, a
`config.toml` value, or any output section 1 to 5 has already produced.

## Q1 — should the README state the null in its first paragraph?

**Yes.**

Step 7.4's README states the null in its **first paragraph**, before the four
questions are worked through:

- the filtered classifier's labels move mostly at refits,
- it separates factor premia no better than chance,
- and no cell of the timing grid is distinguishable from zero.

`PLAN.md` step 7.4 is amended to say so. This goes further than the section 4
Q1 answer, which fixed the *framing* around the null but left the null to
emerge as the four questions were answered in turn. A reader who stops after
the first paragraph must already have the result.

## Q2 — `min_regime_obs` at 24 holds 28% of the window in fallback

**Accepted as a property of the design, with a reporting requirement.**

The 24-month minimum is not a defect to be tuned away. It is what stops a
weight being set from a handful of months in a regime the strategy has barely
seen, and the cost of that protection — the strategy being definitionally the
comparator in those months — is the price of it.

What is added is the reporting that lets a reader tell the two apart. `PLAN.md`
step 6.4 also reports, **per `min_regime_obs` setting**, beside the `diff`:

- the number and the share of out-of-sample months in fallback,
- the mean absolute weight deviation from 1/5.

A larger `diff` at 12 is read as **more trading, not more signal**, unless the
fallback share says otherwise. The status file flagged in advance that 12 would
move the `diff` in one direction or the other mechanically; with the fallback
share and the deviation printed next to it, that claim is checkable instead of
asserted.

## Q3 — nothing in the plan catches leave-one-month-out fragility

**Yes, and it becomes step 6.7.**

The headline `diff` flips sign when 2026-07-31 is dropped from 257 months, and
section 4's one pairwise difference that excludes zero leans on 2009-03. There
was no leave-one-month-out or trimmed statistic anywhere in `PLAN.md`, and step
6.5 sweeps only `block_size`. A new **step 6.7 — fragility of the headline
statistics** is added to `PLAN.md`, building `regime/robustness.py`'s
`leave_one_month_out`, `trimmed_diff` and `leave_one_month_out_pairwise`, and
applying both statistics to the headline timing cell and to the one section 4
pairwise difference that excludes zero (`hmm_filtered`, UMD, state 0 against
state 2).

Step 6.7 is added to `run.long_steps` only if it exceeds 20 minutes; the
session measures first.

## What this file does not do

It does not reopen `decisions/primary_feature_set.md`, does not change
`config.toml`, and does not cause any section 1 to 5 output to be regenerated
with different content. The headline cell stays where `config.toml` fixed it
before section 5 ran: η 0.5, lag 1, 20 bp, `hmm_filtered`.

Every robustness run of section 6 is reported **against** that headline cell.
None of them replaces it, and no sentence of a review or status file may
present a robustness variant as the project's result, whatever it shows.
