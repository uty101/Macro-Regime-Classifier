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
    drift rather than as relabelling. Whether that slot holds the same regime
    from refit to refit is the separate question ``anchor_agreement.csv``
    answers.
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
