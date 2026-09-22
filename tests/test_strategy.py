"""Section 5 — the trailing conditional Sharpe, the weights, the backtest, the grid and the timing-gain bootstrap."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from regime.config import load_config
from regime.strategy import (
    _weight_row,
    backtest,
    static_weights,
    trailing_conditional_sharpe,
    weights,
)

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


def test_fallback_unassigned():
    """An unassigned date has no regime to condition on, so it holds the static vector."""
    cfg = _cfg(strategy_min_regime_obs=1)
    earned = [0.05, 0.04, 0.06, 0.03, 0.05, 0.04]
    label_frame, factors = _frames([0] * 6, earned)
    label_frame.iloc[-1, label_frame.columns.get_loc("assigned")] = False

    w = weights(label_frame, factors, 1.0, cfg)

    last = w.iloc[-1]
    assert list(last) == pytest.approx([0.2] * 5, abs=1e-12)
    assert w.attrs["fallback_branch"].iloc[-1] == "unassigned"


def test_fallback_below_min_regime_obs():
    """n_k below strategy.min_regime_obs is the static vector however good the trailing Sharpe."""
    cfg = _cfg(strategy_min_regime_obs=24)
    earned = [0.05, 0.04, 0.06, 0.03, 0.05, 0.04]
    label_frame, factors = _frames([0] * 6, earned)

    w = weights(label_frame, factors, 1.0, cfg)

    assert np.allclose(w.to_numpy(), 0.2, atol=1e-12)
    assert set(w.attrs["fallback_branch"]) == {"thin_regime"}


def test_fallback_no_positive_sharpe():
    """Every factor losing money in the regime leaves nothing to tilt towards."""
    cfg = _cfg(strategy_min_regime_obs=2)
    earned = [-0.05, -0.04, -0.06, -0.03, -0.05, -0.04]
    label_frame, factors = _frames([0] * 6, earned)

    w = weights(label_frame, factors, 1.0, cfg)

    branch = w.attrs["fallback_branch"]
    assert np.allclose(w.loc[branch == "no_positive_sharpe"].to_numpy(), 0.2, atol=1e-12)
    assert (branch.iloc[2:] == "no_positive_sharpe").all()          # from the third date every S is negative
    assert list(w.iloc[-1]) == pytest.approx([0.2] * 5, abs=1e-12)


def test_blend_formula_by_hand():
    """eta = 0.5, S = (0.8, -0.2, 0.4, 0.0, NaN) -> the blended vector, to 1e-12.

    S+ is (0.8, 0, 0.4, 0, 0) and sums to 1.2. The negative, the exact zero and
    the NaN all contribute nothing to the tilt and keep only their (1 - eta)/5
    share, so three of the five weights are 0.1 and the two survivors split
    0.5 in the ratio 2:1.
    """
    cfg = _cfg()
    s = pd.Series([0.8, -0.2, 0.4, 0.0, np.nan], index=list(UNIVERSE), dtype="float64")

    row, branch = _weight_row(s, n_k=cfg.strategy_min_regime_obs, assigned=True, eta=0.5, cfg=cfg)

    expected = [0.1 + 0.5 * 0.8 / 1.2, 0.1, 0.1 + 0.5 * 0.4 / 1.2, 0.1, 0.1]
    assert list(row) == pytest.approx(expected, abs=1e-12)
    assert branch == ""
    assert sum(row) == pytest.approx(1.0, abs=1e-12)


def test_weights_sum_to_one():
    """Every row of every eta sums to 1, blended rows and fallback rows alike."""
    cfg = _cfg(strategy_min_regime_obs=6)
    rng = np.random.default_rng(cfg.run_seed)
    earned = list(rng.normal(0.005, 0.03, size=60))
    labels = list(rng.integers(0, 3, size=60))
    label_frame, factors = _frames(labels, earned)
    label_frame.iloc[::9, label_frame.columns.get_loc("assigned")] = False

    for eta in cfg.strategy_eta_grid:
        w = weights(label_frame, factors, eta, cfg)
        assert list(w.columns) == list(UNIVERSE)
        assert list(w.index) == list(label_frame.index)
        assert np.allclose(w.sum(axis=1).to_numpy(), 1.0, atol=1e-12)
        assert (w.to_numpy() >= 0).all()
        assert set(w.attrs["fallback_branch"]) & {"unassigned"}


BACKTEST_INDEX = _months("2020-01-31", 5)
BACKTEST_WEIGHTS = pd.DataFrame(
    [[0.2, 0.2, 0.2, 0.2, 0.2],
     [0.4, 0.1, 0.1, 0.2, 0.2],
     [0.0, 0.0, 0.5, 0.5, 0.0]],
    index=_months("2020-01-31", 3),
    columns=list(UNIVERSE),
)
BACKTEST_FACTORS = pd.DataFrame(
    [[0.00, 0.00, 0.00, 0.00, 0.00],       # 2020-01-31, earned by no decision in this example
     [0.01, 0.02, -0.01, 0.00, 0.03],      # 2020-02-29
     [-0.02, 0.01, 0.04, 0.01, -0.01],     # 2020-03-31
     [0.03, -0.03, 0.02, 0.02, 0.00],      # 2020-04-30
     [0.01, 0.01, 0.01, -0.02, 0.02]],     # 2020-05-31
    index=BACKTEST_INDEX,
    columns=list(UNIVERSE),
)


def test_backtest_three_month_literal():
    """PLAN.md step 5.3's example, both lags, every number written out.

    lag 0 earns month t+1 and lag 1 earns month t+2, so the same three weight
    vectors earn three different months and the turnover charged in a given
    calendar month moves with the lag. The turnover series is identical in
    both -- 0.0, 0.2, 0.7 -- because it is a property of the weight book, not
    of when the book earns.
    """
    cfg = _cfg()

    zero = backtest(BACKTEST_WEIGHTS, BACKTEST_FACTORS, lag=0, cost_bp=20, cfg=cfg)

    assert list(zero.columns) == ["gross_ret", "turnover", "cost", "net_ret"]
    assert list(zero.index) == [pd.Timestamp("2020-02-29"), pd.Timestamp("2020-03-31"), pd.Timestamp("2020-04-30")]
    assert list(zero["gross_ret"]) == pytest.approx([0.0100, -0.0030, 0.0200], abs=1e-12)
    assert list(zero["turnover"]) == pytest.approx([0.0, 0.2, 0.7], abs=1e-12)
    assert list(zero["cost"]) == pytest.approx([0.0000, 0.0004, 0.0014], abs=1e-12)
    assert list(zero["net_ret"]) == pytest.approx([0.0100, -0.0034, 0.0186], abs=1e-12)

    one = backtest(BACKTEST_WEIGHTS, BACKTEST_FACTORS, lag=1, cost_bp=20, cfg=cfg)

    assert list(one.index) == [pd.Timestamp("2020-03-31"), pd.Timestamp("2020-04-30"), pd.Timestamp("2020-05-31")]
    assert list(one["gross_ret"]) == pytest.approx([0.0060, 0.0150, -0.0050], abs=1e-12)
    assert list(one["turnover"]) == pytest.approx([0.0, 0.2, 0.7], abs=1e-12)
    assert list(one["cost"]) == pytest.approx([0.0000, 0.0004, 0.0014], abs=1e-12)
    assert list(one["net_ret"]) == pytest.approx([0.0060, 0.0146, -0.0064], abs=1e-12)


def test_backtest_drops_decision_dates_without_an_earning_month():
    """At lag 1 the last decision date of a five-month factor file has no month t+2."""
    cfg = _cfg()
    w = pd.DataFrame(0.2, index=BACKTEST_INDEX, columns=list(UNIVERSE))

    assert list(backtest(w, BACKTEST_FACTORS, lag=0, cost_bp=0, cfg=cfg).index) == list(BACKTEST_INDEX[1:])
    assert list(backtest(w, BACKTEST_FACTORS, lag=1, cost_bp=0, cfg=cfg).index) == list(BACKTEST_INDEX[2:])


def test_static_weights_have_zero_turnover():
    """A book that never leaves 1/5 pays no cost, including on its first row."""
    cfg = _cfg()
    index = _months("2005-01-31", 30)
    factors = pd.DataFrame(
        np.random.default_rng(cfg.run_seed).normal(0.005, 0.03, size=(31, 5)),
        index=_months("2005-01-31", 31), columns=list(UNIVERSE),
    )
    w = static_weights(index, cfg)

    for lag in cfg.strategy_lag_grid:
        for cost_bp in cfg.strategy_cost_bp_grid:
            result = backtest(w, factors, lag=lag, cost_bp=cost_bp, cfg=cfg)
            assert np.allclose(result["turnover"].to_numpy(), 0.0, atol=1e-15)
            assert np.allclose(result["cost"].to_numpy(), 0.0, atol=1e-15)
            assert np.allclose(result["net_ret"].to_numpy(), result["gross_ret"].to_numpy(), atol=1e-15)
