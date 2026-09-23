# OPEN

Items that `PLAN.md`, `CLAUDE.md` and `config.toml` leave genuinely
unspecified. Each carries two concrete options. A session never picks one
(rule 1); it finishes every step that does not depend on the item and reports
it in that section's review file.

---

## 1. K = 5 cannot complete the expanding protocol (step 6.1) — RESOLVED, Option A

**Raised:** session 6, step 6.1.

**What happens.** `run_expanding_hmm` at K = 5 on the primary d = 8 feature set
fails at the **2006-12-31** refit. hmmlearn fits it, but the fitted covariance
of one state is numerically singular, and `forward_filter` refuses it:

```
numpy.linalg.LinAlgError: When `allow_singular is False`, the input matrix
must be symmetric positive definite.
```

State 4's covariance at that refit has eigenvalues

```
2.170e-16  2.506e-03  7.736e-03  8.326e-02  1.292e-01  7.726e-01  1.629e+00  3.379e+00
```

— a condition number of 1.6e16 on a matrix that is symmetric to 2.2e-16. The
state has collapsed onto a 7-dimensional hyperplane of the 8-dimensional
feature space. It is over-parameterisation, not a numerical accident: a K = 5,
d = 8 full-covariance HMM has `n_params(5, 8) = 240` free parameters and that
refit trains on **192 rows**. `hmm.min_covar` (1e-3) floors the *diagonal* and
cannot prevent a rank deficiency in a full covariance matrix.

K = 2, K = 3 and K = 4 complete. `select_k` also completes at K = 5, because it
fits only the 168-row first window and never filters forward with those
parameters — which is why this did not surface in section 3.

**Why it is open.** `PLAN.md` step 6.1 says "for each K in `cfg.hmm_k_grid`"
and says nothing about a K that cannot be filtered. Both answers below are
defensible and each changes what the project reports.

**Option A — report K = 5 as a failure of the variant and leave the code
alone.** Step 6.1 covers K ∈ {2, 3, 4}; `outputs/tables/robustness/k_variant_outcomes.csv`
carries K = 5 with `status = singular_covariance` and the parameter count
beside it, and the review file and README name it. `hmm.k_grid` is unchanged,
so `bic_by_k.csv` still ranks all four. The cost: step 6.1 covers three K
values rather than four, and the robustness of the K choice is tested over a
narrower range than planned.

**Option B — pass `allow_singular=True` in `regime/models/hmm_numpy.py::log_emissions`.**
scipy then evaluates the collapsed Gaussian through a pseudo-inverse and K = 5
completes. The cost: this is a change to the *main run's* filter, not to a
variant — it is the same function the published K = 3 classifier uses. K = 3's
covariances are well conditioned, so the K = 3 numbers would not move, but the
filter would stop refusing a degenerate state anywhere, and a density
evaluated on a rank-deficient covariance is not comparable across states. A
collapsed state would be filtered rather than reported.

**What session 6 did:** neither. K ∈ {2, 3, 4} are complete and reported; K = 5
is recorded in `k_variant_outcomes.csv` and in `review/section_6.md`. Nothing
else in section 6 depends on it.

**Resolved (session 7, reviewer decision): Option A.** K = 5 is reported as a
variant that cannot be filtered; `regime/models/hmm_numpy.py` is unchanged. The
reason: 240 free parameters on 192 training rows collapses a state onto a
7-dimensional hyperplane, and `forward_filter` refusing a singular covariance
is correct behaviour. The failure is the finding. See
`decisions/section_6_review.md`, Q1. This item is kept, not deleted.

**Addendum (session 9): the same rank deficiency surfaces as one of two
exceptions depending on the library stack.** On the stack this repo was built
on, scipy's positive-definite check raises
`numpy.linalg.LinAlgError`, as quoted above. On another stack hmmlearn's own
`_utils._validate_covars` rejects the fitted covariance first and raises
`ValueError("component 4 of 'full' covars must be symmetric, positive-definite")`
before `forward_filter` is ever reached. Both are the same finding — a state
collapsed onto a hyperplane — and step 8.2 made `is_singular_covariance`
recognise both, so K = 5 is recorded as `status = singular_covariance` in
`k_variant_outcomes.csv` either way rather than aborting the run on one stack
and not the other. The resolution is unchanged: Option A, report the failure,
no pseudo-inverse, nothing in `regime/models/hmm_numpy.py` moves.
