"""Growth/inflation rules quadrant labels (``rules_labels``). Built in step 3.1.

The baseline classifier the HMM is judged against. Two raw features, each
compared to its own real-time expanding median under the same first-window
rule as standardisation (convention 2): inside the first window every row is
compared to the median of the whole window, which is in-sample by construction
and is the window the first HMM is fitted on; after it the median at t uses
every decision date from ``features_from`` to t inclusive and nothing later.

Raw features, not z, because a median is scale-free: standardising first would
subtract an expanding mean and divide by an expanding standard deviation and
then take a median of the result, which is the same ordering with two extra
moving parts.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from regime.config import Config

GROWTH_DOWN_INFLATION_DOWN = 0
GROWTH_UP_INFLATION_DOWN = 1
GROWTH_DOWN_INFLATION_UP = 2
GROWTH_UP_INFLATION_UP = 3


def expanding_median(s: pd.Series, cfg: Config) -> pd.Series:
    """The real-time median of ``s`` at each decision date under the first-window rule.

    For t <= ``first_window_end`` the median of the non-NaN rows of
    ``[features_from, first_window_end]``; after it the median of the non-NaN
    rows of ``[features_from, t]`` inclusive. NaN rows contribute nothing and
    are not filled.
    """
    start = pd.Timestamp(cfg.sample_features_from)
    window_end = pd.Timestamp(cfg.sample_first_window_end)
    x = s.loc[s.index >= start]
    med = x.expanding().median()
    in_window = x.index <= window_end
    med.loc[in_window] = x.loc[in_window].median()
    return med


def rules_labels(raw: pd.DataFrame, cfg: Config) -> pd.Series:
    """The quadrant label at every decision date where both rules features are present.

    ``growth_up = raw[cfg.rules_growth_feature] > expanding_median`` and
    likewise for inflation; label = 0 down/down, 1 up/down, 2 down/up, 3 up/up.
    Index: the rows of ``raw`` from ``cfg.sample_features_from`` where both
    features are non-NaN. Written to ``data/processed/rules_labels.parquet``.
    """
    growth, inflation = cfg.rules_growth_feature, cfg.rules_inflation_feature
    start = pd.Timestamp(cfg.sample_features_from)
    x = raw.loc[raw.index >= start, [growth, inflation]]

    growth_up = x[growth] > expanding_median(x[growth], cfg)
    inflation_up = x[inflation] > expanding_median(x[inflation], cfg)
    labels = (growth_up.astype(int) + 2 * inflation_up.astype(int)).loc[x.notna().all(axis=1)]

    labels = labels.astype("int64")
    labels.name = "rules_label"
    labels.index.name = "date"
    path = Path(cfg.outputs_processed_dir) / "rules_labels.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_frame().to_parquet(path)
    return labels
