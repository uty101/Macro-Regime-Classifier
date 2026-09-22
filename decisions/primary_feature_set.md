# Decision — the primary feature set

**Outcome: `features.primary = "core"` (d = 8, `dgs10_level` included).** Applied
mechanically by `regime.tables.primary_feature_set_decision` to
`outputs/tables/classifier_diagnostics.csv`; the value in `config.toml` is
unchanged from the default, and no output was regenerated as a result.

## The rule, quoted

From `instructions/03b_section_3_revision.md`, item 2, fixed before the
diagnostics below existed and before any factor return was loaded:

> Decision rule, applied mechanically: primary becomes "core_no_level" if and
> only if the d = 7 row has detects_2008 true AND detects_2020 true AND
> n_changes_on_refit_dates strictly below the d = 8 row's. Otherwise primary
> stays "core".

## The diagnostics it was applied to

```
  feature_set  n_filtered_changes  n_changes_on_refit_dates  share_on_refit_dates  median_run_months  max_expected_duration  n_infinite_durations  n_degenerate_states  max_matched_distance  detects_2008  detects_2020  filtered_smoothed_agreement
         core                  20                         9              0.450000               10.0             129.875370                     1                    0              2.689391         False          True                     0.682171
core_no_level                  21                         7              0.333333                9.5              57.102728                     0                    0              2.131451         False         False                     0.403101
```

## How the rule resolved

The d = 7 row fails the first two conditions, so the third never matters:

| condition | d = 7 value | holds? |
|---|---|---|
| `detects_2008` true | `False` | no |
| `detects_2020` true | `False` | no |
| `n_changes_on_refit_dates` < d = 8's 9 | 7 | yes |

One failed condition is enough. **Primary stays `"core"`.**

## What that does and does not say

It says only that dropping `dgs10_level` did not buy the two things the rule
was written to buy. Removing the feature made the filter *less* responsive to
the two episodes the rule tests, not more: d = 7 holds one state from
2008-02-29 to 2008-11-30 and does not move at all through 2020-03 or 2020-04,
where d = 8 switches at 2020-04-30.

It does not say `dgs10_level` is harmless. Two of its costs are unchanged and
are still on the record:

- **Neither set detects 2008 under this test.** `detects_2008` is `False` for
  d = 8 as well: the filtered label at 2008-11-30 is the same as at
  2008-06-30. The d = 8 filter does move, but a month later — at 2008-12-31,
  which is a refit date. This is the observation that prompted the revision and
  it survives it.
- **9 of d = 8's 20 label changes still land exactly on a refit date** (45%).
  Chaining improved this — the sorted run had 15 of 26, or 58% — but did not
  remove it.

The d = 7 run is kept in full under `outputs/regimes/robustness/nolevel/` and
`outputs/tables/robustness/nolevel/`, and is what step 6.6 carries into the
conditional statistics and the timing grid. If the reviewer wants the question
reopened, the material is there; reopening it is a change to this decision
file and to convention 15, not something a session decides on its own.

## Provenance

- Rule fixed: `instructions/03b_section_3_revision.md` (committed before any
  code in this revision was written).
- Diagnostics produced by: `python -m regime.run --section 3`, which runs both
  feature sets every time, so the table can never go stale against the config
  key it justifies.
- Applied by: `regime.tables.primary_feature_set_decision`, tested in
  `tests/test_tables.py::test_primary_feature_set_rule_needs_all_three_conditions`.
- No factor return was loaded in the session that made this decision.
