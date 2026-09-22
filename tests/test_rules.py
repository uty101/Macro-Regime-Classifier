"""Step 3.1: the quadrant labels and the first-window median rule.

Both tests build their own raw frame and compute the medians independently of
``regime.models.rules`` — by hand in the first, with ``np.median`` on explicit
slices in the second — so a change to the expanding rule cannot pass by
agreeing with itself.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

from regime.config import load_config
from regime.models.rules import rules_labels

ROOT = Path(__file__).resolve().parents[1]


def _cfg(tmp_path, index, first_window_end):
    """The real config with the sample dates moved onto ``index`` and outputs in tmp_path."""
    cfg = load_config(str(ROOT / "config.toml"))
    return dataclasses.replace(
        cfg,
        sample_features_from=str(index[0].date()),
        sample_first_window_end=str(pd.Timestamp(first_window_end).date()),
        outputs_processed_dir=str(tmp_path),
    )


def test_quadrant_labels_by_hand(tmp_path) -> None:
    # Nine rows, all inside the first window, so every row is compared to the
    # median of the whole window: 5 for growth and 5 for inflation, by hand.
    index = pd.date_range("2000-01-31", periods=9, freq="ME")
    raw = pd.DataFrame(
        {
            "indpro_chg12": [1.0, 2, 3, 4, 5, 6, 7, 8, 9],
            "cpi_3m_ann": [1.0, 9, 2, 8, 3, 7, 4, 6, 5],
        },
        index=index,
    )
    raw.index.name = "date"
    cfg = _cfg(tmp_path, index, index[-1])
    assert (cfg.rules_growth_feature, cfg.rules_inflation_feature) == ("indpro_chg12", "cpi_3m_ann")

    labels = rules_labels(raw, cfg)

    assert labels.name == "rules_label"
    assert len(labels) == 9
    # growth median 5, inflation median 5; ">" is strict, so 5 is "down".
    assert labels.loc[index[0]] == 0   # growth 1 down, inflation 1 down
    assert labels.loc[index[8]] == 1   # growth 9 up,   inflation 5 down
    assert labels.loc[index[1]] == 2   # growth 2 down, inflation 9 up
    assert labels.loc[index[7]] == 3   # growth 8 up,   inflation 6 up
    assert (labels.to_frame() == pd.read_parquet(Path(tmp_path) / "rules_labels.parquet")).all().all()


def test_first_window_median_rule(tmp_path) -> None:
    # Ten window rows then four more. Three low growth rows after the window
    # pull the expanding median down from 14.5 to 13.5, so the last row
    # (growth 14.0) is "up" under the expanding rule and "down" under the
    # frozen window median. That row is what tells the two rules apart.
    index = pd.date_range("2000-01-31", periods=14, freq="ME")
    growth = np.array([10.0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1, 2, 3, 14.0])
    inflation = np.array([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    raw = pd.DataFrame({"indpro_chg12": growth, "cpi_3m_ann": inflation}, index=index)
    raw.index.name = "date"
    cfg = _cfg(tmp_path, index, index[9])

    labels = rules_labels(raw, cfg)

    window_growth_median = float(np.median(growth[:10]))            # 14.5
    window_inflation_median = float(np.median(inflation[:10]))      # 5.5

    # A row inside the window is compared to the median of the whole window.
    inside = 2                                                       # growth 12, inflation 3
    expected_inside = int(growth[inside] > window_growth_median) + 2 * int(
        inflation[inside] > window_inflation_median
    )
    assert expected_inside == 0
    assert labels.loc[index[inside]] == expected_inside

    # A row after the window is compared to the median of rows 0 .. that row.
    after = 13                                                       # growth 14.0, inflation 14
    expanding_growth_median = float(np.median(growth[: after + 1]))  # 13.5
    expanding_inflation_median = float(np.median(inflation[: after + 1]))  # 7.5
    assert (expanding_growth_median, expanding_inflation_median) == (13.5, 7.5)
    expected_after = int(growth[after] > expanding_growth_median) + 2 * int(
        inflation[after] > expanding_inflation_median
    )
    assert expected_after == 3
    assert labels.loc[index[after]] == expected_after

    # The frozen window median would have called that row 2, not 3: the two
    # rules genuinely differ, so this test can tell them apart.
    frozen_after = int(growth[after] > window_growth_median) + 2 * int(
        inflation[after] > window_inflation_median
    )
    assert frozen_after == 2 != expected_after

    # And every row, not just those two, follows the rule in force at it.
    expected = [
        int(growth[i] > (window_growth_median if i <= 9 else float(np.median(growth[: i + 1]))))
        + 2 * int(inflation[i] > (window_inflation_median if i <= 9 else float(np.median(inflation[: i + 1]))))
        for i in range(14)
    ]
    assert list(labels.to_numpy()) == expected


def test_rows_with_a_nan_feature_are_absent(tmp_path) -> None:
    index = pd.date_range("2000-01-31", periods=6, freq="ME")
    raw = pd.DataFrame(
        {"indpro_chg12": [1.0, 2, np.nan, 4, 5, 6], "cpi_3m_ann": [1.0, 2, 3, np.nan, 5, 6]},
        index=index,
    )
    raw.index.name = "date"
    labels = rules_labels(raw, _cfg(tmp_path, index, index[-1]))
    assert list(labels.index) == [index[0], index[1], index[4], index[5]]
