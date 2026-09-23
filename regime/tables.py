"""Output tables and regime_labels.csv. feature_sanity is step 2.4, the HMM tables step 3.7, the rest step 7.2."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from regime.config import Config

FEATURE_SANITY_COLUMNS = ("feature", "year", "min", "max", "n_nan")


def feature_sanity(raw: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """One row per feature x calendar year of ``raw``: ``feature, year, min, max, n_nan``.

    Features in the order of ``cfg.features_core`` then ``cfg.features_robustness``;
    years ascending within each feature. ``min`` and ``max`` ignore NaN and are
    NaN for a year with no valid value; ``n_nan`` counts the NaN rows of that
    feature in that year. Written to ``outputs/tables/feature_sanity.csv``.
    """
    features = list(cfg.features_core) + list(cfg.features_robustness)
    years = raw.index.year
    rows = []
    for feature in features:
        s = raw[feature]
        for year, chunk in s.groupby(years):
            rows.append(
                {
                    "feature": feature,
                    "year": int(year),
                    "min": chunk.min(),
                    "max": chunk.max(),
                    "n_nan": int(chunk.isna().sum()),
                }
            )
    table = pd.DataFrame(rows, columns=list(FEATURE_SANITY_COLUMNS))
    out = Path(cfg.outputs_tables_dir) / "feature_sanity.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    return table


def expected_durations(transmat: np.ndarray) -> np.ndarray:
    """``1 / (1 - A_kk)`` per state: the mean run length of a geometric holding time.

    A self-transition probability of exactly 1 gives ``inf``, which is the
    correct answer for an absorbing state and is reported rather than clipped.
    """
    diag = np.diag(np.asarray(transmat, dtype="float64"))
    with np.errstate(divide="ignore"):
        return 1.0 / (1.0 - diag)


def write_hmm_tables(params: list, feature_names: list[str], cfg: Config) -> None:
    """Per-refit transition matrices, ``expected_duration.csv`` and ``param_drift.csv``.

    All three are written from anchored parameters, so state k is the same
    slot in every row — which is what makes ``param_drift.csv`` readable as
    drift rather than as relabelling. Under chained numbering (convention 16)
    that slot follows the nearest state from refit to refit; how far it had to
    reach each year is ``matched_distance`` in ``anchor_chain.csv``.
    """
    tables_dir = Path(cfg.outputs_tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    states = list(range(params[0].K))

    durations, drift = [], []
    for p in params:
        transition = pd.DataFrame(
            p.transmat, index=pd.Index(states, name="from_state"), columns=[f"to_{k}" for k in states]
        )
        transition.to_csv(tables_dir / f"transition_matrix_{p.refit_date:%Y-%m-%d}.csv")
        for k, duration in zip(states, expected_durations(p.transmat)):
            durations.append(
                {"refit_date": p.refit_date.date(), "state": k, "expected_duration": duration}
            )
        for k in states:
            variances = np.diag(p.covars[k])
            for j, feature in enumerate(feature_names):
                drift.append(
                    {
                        "refit_date": p.refit_date.date(),
                        "state": k,
                        "feature": feature,
                        "mean": float(p.means[k, j]),
                        "variance": float(variances[j]),
                    }
                )

    pd.DataFrame(durations, columns=["refit_date", "state", "expected_duration"]).to_csv(
        tables_dir / "expected_duration.csv", index=False
    )
    pd.DataFrame(drift, columns=["refit_date", "state", "feature", "mean", "variance"]).to_csv(
        tables_dir / "param_drift.csv", index=False
    )


CLASSIFIER_DIAGNOSTIC_COLUMNS = (
    "feature_set", "n_filtered_changes", "n_changes_on_refit_dates", "share_on_refit_dates",
    "median_run_months", "max_expected_duration", "n_infinite_durations",
    "n_degenerate_states", "max_matched_distance", "detects_2008", "detects_2020",
    "filtered_smoothed_agreement",
)

DIAGNOSTICS_FROM = "2005-01-31"


def label_runs(labels: pd.Series) -> pd.DataFrame:
    """Consecutive runs of a constant label: columns ``start, state, length``."""
    values = labels.to_numpy()
    breaks = np.flatnonzero(np.r_[True, values[1:] != values[:-1]])
    lengths = np.diff(np.r_[breaks, len(values)])
    return pd.DataFrame(
        {
            "start": labels.index[breaks],
            "state": values[breaks].astype(int),
            "length": lengths.astype(int),
        }
    )


def changes_on_refit_dates(filtered_labels: pd.DataFrame, refits) -> tuple[int, int, list]:
    """``(n_changes, n_on_refit, the dates)`` of filtered label changes from ``DIAGNOSTICS_FROM``.

    A label change that lands exactly on a refit date is a change the
    classifier made because it was refitted, not because the data moved. The
    first row of the frame has no predecessor and never counts as a change.

    Section 6's robustness variants report this beside their timing number so
    the classifier's behaviour travels with it, which is why it lives here
    rather than inside ``classifier_diagnostics_row``.
    """
    labels = filtered_labels["label"]
    changed = labels.ne(labels.shift())
    changed.iloc[0] = False
    change_dates = changed.index[changed.to_numpy()]
    change_dates = change_dates[change_dates >= pd.Timestamp(DIAGNOSTICS_FROM)]
    refit_set = {pd.Timestamp(d) for d in refits}
    on_refit = [d for d in change_dates if d in refit_set]
    return len(change_dates), len(on_refit), on_refit


def classifier_diagnostics_row(
    feature_set: str,
    filtered_labels: pd.DataFrame,
    smoothed_labels: pd.DataFrame,
    refits: list,
    durations: pd.DataFrame,
    state_counts: pd.DataFrame,
    chain_table: pd.DataFrame,
) -> dict:
    """One row of ``classifier_diagnostics.csv`` for one feature set.

    Every count of label changes is taken from ``DIAGNOSTICS_FROM`` (2005-01-31)
    so the first out-of-sample date, which has no predecessor, is excluded.
    ``median_run_months`` is over the whole filtered series, including that
    first date's run.

    ``n_changes_on_refit_dates`` is the diagnostic the revision exists for: a
    label change that lands exactly on a refit date is a change the classifier
    made because it was refitted, not because the data moved.
    """
    labels = filtered_labels["label"]
    n_changes, n_on_refit, _dates = changes_on_refit_dates(filtered_labels, refits)

    finite = durations.loc[np.isfinite(durations["expected_duration"]), "expected_duration"]
    window = filtered_labels.index >= pd.Timestamp(DIAGNOSTICS_FROM)
    shared = filtered_labels.index[window].intersection(smoothed_labels.index)

    def label_at(date: str) -> int | None:
        stamp = pd.Timestamp(date)
        return int(labels.loc[stamp]) if stamp in labels.index else None

    return {
        "feature_set": feature_set,
        "n_filtered_changes": n_changes,
        "n_changes_on_refit_dates": n_on_refit,
        "share_on_refit_dates": (n_on_refit / n_changes) if n_changes else float("nan"),
        "median_run_months": float(label_runs(labels)["length"].median()),
        "max_expected_duration": float(finite.max()) if len(finite) else float("nan"),
        "n_infinite_durations": int((~np.isfinite(durations["expected_duration"])).sum()),
        "n_degenerate_states": int(state_counts["degenerate"].sum()),
        "max_matched_distance": float(chain_table["matched_distance"].max()),
        "detects_2008": label_at("2008-11-30") != label_at("2008-06-30"),
        "detects_2020": label_at("2020-04-30") != label_at("2019-12-31"),
        "filtered_smoothed_agreement": float(
            (filtered_labels.loc[shared, "label"] == smoothed_labels.loc[shared, "label"]).mean()
        ),
    }


def primary_feature_set_decision(diagnostics: pd.DataFrame) -> str:
    """The pre-registered rule, applied mechanically to ``classifier_diagnostics.csv``.

    Fixed in ``instructions/03b_section_3_revision.md`` before the diagnostics
    existed, and quoted verbatim in ``decisions/primary_feature_set.md``:

        primary becomes "core_no_level" if and only if the d = 7 row has
        detects_2008 true AND detects_2020 true AND n_changes_on_refit_dates
        strictly below the d = 8 row's. Otherwise primary stays "core".

    No other column of the table enters the decision. The rest are reported so
    the reviewer can see what the rule did not weigh.
    """
    rows = diagnostics.set_index("feature_set")
    d7, d8 = rows.loc["core_no_level"], rows.loc["core"]
    chooses_d7 = (
        bool(d7["detects_2008"])
        and bool(d7["detects_2020"])
        and int(d7["n_changes_on_refit_dates"]) < int(d8["n_changes_on_refit_dates"])
    )
    return "core_no_level" if chooses_d7 else "core"
