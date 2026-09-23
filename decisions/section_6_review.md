# Decision — the reviewer's answers to the section 6 questions

The questions are those in `instructions/06_section_6.status.md`, "Questions
and blockers for the reviewer". The reviewer answered them in the session 7
instruction message (`instructions/07_section_7.md`); this file records the
answers so nothing in section 7 has to go back to chat for them.

None of these answers changes a convention, a `config.toml` value, or any
output sections 1 to 6 have already produced.

## Q1 — `decisions/OPEN.md` item 1, K = 5 cannot complete the expanding protocol

**Option A. K = 5 is reported as a variant that cannot be filtered; no change
to `regime/models/hmm_numpy.py`.**

The reason: 240 free parameters on 192 training rows collapses a state onto a
7-dimensional hyperplane, and `forward_filter` refusing a singular covariance
is correct behaviour. **The failure is the finding.** A filter that quietly
evaluated a density on a rank-deficient covariance — Option B — would be
trading a reported failure for an unreported one, on the main run's own code
path.

Step 6.1 therefore covers K ∈ {2, 3, 4} as built, with K = 5 carried in
`outputs/tables/robustness/k_variant_outcomes.csv` as
`status = singular_covariance` beside its parameter count. `hmm.k_grid` is
unchanged, so `bic_by_k.csv` still ranks all four. `decisions/OPEN.md` item 1
is marked resolved with this choice and this reasoning, and is **not** deleted.

## Q2 — the two definitions step 6.7 fixed by implementation

**Both accepted as implemented.**

- The trim count is `floor(0.05 n)` — a floor, so "the 5% of months" never
  removes more than 5% of them.
- Pairwise influence is measured by how far a month's removal moves the
  statistic (`loo_diff − full_diff`), because the two states occupy **disjoint**
  months and no per-month timed-minus-static pairing exists for them.

Both are stated in the README's methods note, so a reader of the finished repo
meets them where the numbers are quoted rather than in the source.

## Q3 — `covars_for_setter` touches main-run code

**The fix is accepted, and flagging a change to main-run code was the right
call.** `"full"` returns its input unchanged, the `k3` variant still reproduces
the main headline row to every digit, and a section 6 rerun left `git status`
clean — so the published K = 3 classifier's numbers are demonstrably unmoved.
Raising it anyway is what a reviewer needs to be told, not what a session
should have decided quietly.

## Q4 — the two findings that are not about timing

Both belong in step 7.4's "What did not work", as the status file proposed: the
`10feat` variant's 20 of 51 chained states matching at more than 1 z, and the
anchor-versus-sort disagreement growing with K.

## What this file does not do

It does not reopen `decisions/primary_feature_set.md`, does not change
`config.toml`, does not change `regime/models/hmm_numpy.py`, and does not cause
any section 1 to 6 output to be regenerated with different content. The
headline cell stays where `config.toml` fixed it before section 5 ran: η 0.5,
lag 1, 20 bp, `hmm_filtered`.
