"""Section 5 — the trailing conditional Sharpe, the weights, the backtest, the grid and the timing-gain bootstrap."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from regime.config import load_config
from regime.strategy import trailing_conditional_sharpe

UNIVERSE = ("SMB", "HML", "RMW", "CMA", "UMD")


def _months(start: str, n: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="ME", name="date")


def _cfg(**kwargs):
    """The real config with the allocation universe as the factor set, so the frames stay small."""
    base = dataclasses.replace(
        load_config(), strategy_factors=UNIVERSE, strategy_universe=UNIVERSE,
        sample_first_window_end="2005-01-31",
    )
    return dataclasses.replace(base, **kwargs)


def _frames(labels, returns, start="2005-01-31"):
    """A label frame over n decision dates and a factor frame over n+1 months.

    ``returns[i]`` is the return of month ``i + 1`` of the factor index, i.e.
    the return the decision at date ``i`` earns. The factor frame therefore
    carries one leading month that no decision earns and one trailing month
    that the last decision does.
    """
    n = len(labels)
    label_frame = pd.DataFrame({"label": labels, "assigned": [True] * n}, index=_months(start, n))
    factors = pd.DataFrame(
        {f: [0.0] + list(returns) for f in UNIVERSE}, index=_months(start, n + 1)
    )
    return label_frame, factors


def test_trailing_sharpe_by_hand():
    """Literal expected values on a five-row example.

    Decision dates 2005-01-31 .. 2005-05-31 with labels 0, 1, 0, 0, 1; the
    returns they earn are the months 2005-02 .. 2005-06. At t = 2005-05-31 the
    sample for k = label_t = 1 is the single date 2005-02-28, so n_k is 1 and
    the Sharpe is NaN; the sample for k = 0 at the same t would be three dates.
    """
    cfg = _cfg()
    earned = [0.01, -0.02, 0.03, 0.005, -0.01]
    labels = [0, 1, 0, 0, 1]
    label_frame, factors = _frames(labels, earned)

    t = pd.Timestamp("2005-05-31")
    s, n_k = trailing_conditional_sharpe(label_frame, factors, t, cfg)

    assert n_k == 1                                   # only 2005-02-28 carries label 1 before t
    assert list(s.index) == list(UNIVERSE)
    assert s.isna().all()                             # one observation has no sample standard deviation

    # the same t under k = 0: three dates, 2005-01-31, 2005-03-31 and 2005-04-30
    zero_at_t = label_frame.copy()
    zero_at_t.loc[t, "label"] = 0
    s0, n0 = trailing_conditional_sharpe(zero_at_t, factors, t, cfg)

    assert n0 == 3
    sample = np.array([0.01, 0.03, 0.005])
    expected = sample.mean() / sample.std(ddof=1) * np.sqrt(12.0)
    for f in UNIVERSE:
        assert s0[f] == pytest.approx(expected, abs=1e-12)


def test_trailing_sharpe_is_strictly_before_t():
    """A 1e3 return in month t+1 or t+2 cannot move S or n_k at t; one in month t must.

    Month t+1 is earned by the decision at t and month t+2 by the decision at
    t+1; neither decision date is ``< t``, so neither return may enter. Month
    t is earned by the decision at t-1, which is ``< t`` and carries label k
    here, so it must enter. Planting a return four orders of magnitude larger
    than the rest makes any leak impossible to miss.
    """
    cfg = _cfg()
    earned = [0.01, -0.02, 0.03, 0.005, -0.01, 0.02, -0.015, 0.025]
    labels = [0] * 8
    label_frame, factors = _frames(labels, earned)
    t = pd.Timestamp("2005-06-30")                    # the sixth of eight decision dates

    base_s, base_n = trailing_conditional_sharpe(label_frame, factors, t, cfg)
    assert base_n == 5                                # 2005-01-31 .. 2005-05-31

    # month t+1 (2005-07-31) is earned by the decision at t, and month t+2 by the decision at t+1
    for offset in (1, 2):
        planted = factors.copy()
        planted.loc[t + pd.offsets.MonthEnd(offset), :] = 1e3
        s, n_k = trailing_conditional_sharpe(label_frame, planted, t, cfg)
        assert n_k == base_n, offset
        pd.testing.assert_series_equal(s, base_s)

    # month t itself (2005-06-30) is earned by the decision at 2005-05-31, which is < t
    planted = factors.copy()
    planted.loc[t, :] = 1e3
    s, n_k = trailing_conditional_sharpe(label_frame, planted, t, cfg)

    assert n_k == base_n
    assert not np.allclose(s.to_numpy(), base_s.to_numpy())


def test_trailing_sharpe_excludes_unassigned_and_other_states():
    cfg = _cfg()
    earned = [0.01, 99.0, 0.03, -99.0, 0.005, 0.02]
    label_frame, factors = _frames([0, 0, 0, 1, 0, 0], earned)
    label_frame.iloc[1, label_frame.columns.get_loc("assigned")] = False   # the 99.0 month

    s, n_k = trailing_conditional_sharpe(label_frame, factors, pd.Timestamp("2005-06-30"), cfg)

    assert n_k == 3                                   # 2005-01-31, 2005-03-31, 2005-05-31
    sample = np.array([0.01, 0.03, 0.005])
    for f in UNIVERSE:
        assert s[f] == pytest.approx(sample.mean() / sample.std(ddof=1) * np.sqrt(12.0), abs=1e-12)
