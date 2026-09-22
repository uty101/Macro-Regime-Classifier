"""Step 1.3: the 10-day lookback rule for market series at month-end decision dates."""

import numpy as np
import pandas as pd
import pytest

from regime.config import load_config
from regime.data.fred import month_end_from_daily

T = pd.Timestamp("2010-03-31")


def _series(points: dict) -> pd.Series:
    idx = pd.DatetimeIndex(sorted(pd.Timestamp(k) for k in points))
    return pd.Series([points[str(d.date())] for d in idx], index=idx, name="X", dtype="float64")


@pytest.mark.parametrize(
    "points, expected",
    [
        # an observation dated t is used
        ({"2010-03-30": 1.0, "2010-03-31": 2.0}, 2.0),
        # the last observation dated t - 10 days is used
        ({"2010-03-21": 3.0}, 3.0),
        # the last observation dated t - 11 days gives NaN
        ({"2010-03-20": 4.0}, np.nan),
        # a NaN on t with a value on t - 1 uses t - 1
        ({"2010-03-30": 5.0, "2010-03-31": np.nan}, 5.0),
        # the latest in-window observation wins, not the largest
        ({"2010-03-25": 9.0, "2010-03-29": 6.0}, 6.0),
    ],
    ids=["on_t", "t_minus_10", "t_minus_11", "nan_on_t", "latest_wins"],
)
def test_lookback_rule(points, expected) -> None:
    cfg = load_config()
    assert cfg.fred_lookback_days == 10
    daily = _series({**points, "2010-04-15": 100.0})  # keeps the index extending to a month-end after T
    out = month_end_from_daily(daily, pd.Timestamp("2010-03-31"), cfg.fred_lookback_days)
    assert out.index[0] == T and out.index.name == "date"
    got = out.loc[T]
    if np.isnan(expected):
        assert np.isnan(got)
    else:
        assert got == expected


def test_no_future_observation() -> None:
    cfg = load_config()
    base = {"2010-03-29": 1.5, "2010-04-30": 2.5}
    clean = month_end_from_daily(_series(base), pd.Timestamp("2010-03-31"), cfg.fred_lookback_days)
    planted = month_end_from_daily(
        _series({**base, "2010-04-01": 1e6}), pd.Timestamp("2010-03-31"), cfg.fred_lookback_days
    )
    assert clean.loc[T] == 1.5
    assert planted.loc[T] == 1.5
    assert (planted.loc[T] - clean.loc[T]) == 0.0
    assert 1e6 not in planted.to_numpy()  # 2010-04-30 window is 04-20..04-30, so 04-01 is never used


def test_index_runs_from_start_to_last_month_end() -> None:
    cfg = load_config()
    daily = _series({"2009-12-15": 1.0, "2010-02-10": 2.0, "2010-05-14": 3.0})
    out = month_end_from_daily(daily, pd.Timestamp(cfg.sample_start), cfg.fred_lookback_days)
    assert out.index[0] == pd.Timestamp("1990-01-31")
    assert out.index[-1] == pd.Timestamp("2010-04-30")  # last month-end <= last observation 2010-05-14
    assert out.index.is_monotonic_increasing and out.index.is_unique
