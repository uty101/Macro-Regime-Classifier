"""Output tables and regime_labels.csv. feature_sanity is built in step 2.4; the rest in step 7.2."""

from __future__ import annotations

from pathlib import Path

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
