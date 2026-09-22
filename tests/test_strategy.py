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
    fallback_branches,
    fill_timing_gain,
    run_timing_grid,
    static_weights,
    timing_gain_bootstrap,
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
    assert fallback_branches(w).iloc[-1] == "unassigned"


def test_fallback_below_min_regime_obs():
    """n_k below strategy.min_regime_obs is the static vector however good the trailing Sharpe."""
    cfg = _cfg(strategy_min_regime_obs=24)
    earned = [0.05, 0.04, 0.06, 0.03, 0.05, 0.04]
    label_frame, factors = _frames([0] * 6, earned)

    w = weights(label_frame, factors, 1.0, cfg)

    assert np.allclose(w.to_numpy(), 0.2, atol=1e-12)
    assert set(fallback_branches(w)) == {"thin_regime"}


def test_fallback_no_positive_sharpe():
    """Every factor losing money in the regime leaves nothing to tilt towards."""
    cfg = _cfg(strategy_min_regime_obs=2)
    earned = [-0.05, -0.04, -0.06, -0.03, -0.05, -0.04]
    label_frame, factors = _frames([0] * 6, earned)

    w = weights(label_frame, factors, 1.0, cfg)

    branch = fallback_branches(w)
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
        assert set(fallback_branches(w)) & {"unassigned"}


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


def _grid_frames(cfg, n=80):
    """Four label frames over the same dates, as run.py section 5 passes them in."""
    rng = np.random.default_rng(cfg.run_seed)
    earned = list(rng.normal(0.005, 0.03, size=n))
    _, factors = _frames([0] * n, earned)
    frames = {}
    for i, source in enumerate(cfg.strategy_label_sources):
        labels = list(rng.integers(0, 3, size=n))
        frame, _ = _frames(labels, earned)
        frame.iloc[i :: 13, frame.columns.get_loc("assigned")] = False
        frames[source] = frame
    return frames, factors


def test_timing_grid_has_72_rows_and_unique_keys():
    cfg = _cfg(strategy_min_regime_obs=6)
    frames, factors = _grid_frames(cfg)

    grid = run_timing_grid(frames, factors, cfg)

    assert list(grid.columns) == [
        "eta", "lag", "cost_bp", "source", "sharpe_static", "sharpe_timed", "diff",
        "diff_p05", "diff_p95", "p_one_sided", "mean_turnover", "n_months",
    ]
    expected = (
        len(cfg.strategy_eta_grid) * len(cfg.strategy_lag_grid)
        * len(cfg.strategy_cost_bp_grid) * len(cfg.strategy_label_sources)
    )
    assert expected == 72
    assert len(grid) == 72
    assert not grid.duplicated(subset=["eta", "lag", "cost_bp", "source"]).any()
    assert grid[["diff_p05", "diff_p95", "p_one_sided"]].isna().all().all()   # step 5.5 fills these
    assert np.allclose(grid["diff"], grid["sharpe_timed"] - grid["sharpe_static"], atol=1e-12)
    assert (grid["n_months"] > 0).all()


def test_static_row_is_identical_across_sources():
    """For a given (lag, cost_bp) the static comparator is one number, whatever the source or eta.

    The four sources share identical out-of-sample dates, so the 1/5 book they
    are each compared against is the same book earning the same months. If this
    ever failed, a positive ``diff`` could be a difference of samples rather
    than a difference of strategies.
    """
    cfg = _cfg(strategy_min_regime_obs=6)
    frames, factors = _grid_frames(cfg)

    grid = run_timing_grid(frames, factors, cfg)

    for (lag, cost_bp), block in grid.groupby(["lag", "cost_bp"]):
        assert block["sharpe_static"].nunique() == 1, (lag, cost_bp)
        assert block["n_months"].nunique() == 1, (lag, cost_bp)


def test_timed_equals_static_when_all_sharpes_are_equal():
    """Nothing to choose between the factors means no tilt, at every eta.

    All five allocation factors carry the identical return series, so every
    trailing conditional Sharpe is equal across factors in every regime. S+ is
    then flat, ``eta * S+_f / sum_g S+_g`` is ``eta / 5`` whatever eta is, and
    the blend collapses to ``(1 - eta)/5 + eta/5 = 0.2``. The timed net series
    must match the static one exactly -- including the cost line, since a book
    that never moves pays no turnover.
    """
    cfg = _cfg(strategy_min_regime_obs=6)
    rng = np.random.default_rng(cfg.run_seed)
    n = 60
    earned = list(rng.normal(0.008, 0.03, size=n))
    labels = list(rng.integers(0, 3, size=n))
    label_frame, factors = _frames(labels, earned)

    for eta in cfg.strategy_eta_grid:
        w = weights(label_frame, factors, eta, cfg)
        assert np.allclose(w.to_numpy(), 0.2, atol=1e-12), eta
        # the blend branch was actually reached; this is not a test of the fallbacks
        assert (fallback_branches(w) == "").any(), eta

        for lag in cfg.strategy_lag_grid:
            for cost_bp in cfg.strategy_cost_bp_grid:
                timed = backtest(w, factors, lag, cost_bp, cfg)
                static = backtest(static_weights(w.index, cfg), factors, lag, cost_bp, cfg)
                assert np.allclose(timed["net_ret"].to_numpy(), static["net_ret"].to_numpy(), atol=1e-12)
                assert np.allclose(timed["turnover"].to_numpy(), 0.0, atol=1e-12)


def test_timing_gain_identical_series():
    """A timed book identical to the static one has a gain of exactly 0 and never beats it.

    Both series travel in the same bootstrap row, so every replication's
    difference is 0 rather than merely centred on 0, and both percentiles are
    0. ``p_one_sided`` counts replications with ``diff <= 0``, so an exactly
    zero gain scores 1.0 -- the strongest possible statement that there is no
    gain here.
    """
    cfg = _cfg(bootstrap_n_replications=50)
    rng = np.random.default_rng(cfg.run_seed)
    index = _months("2005-01-31", 60)
    series = pd.Series(rng.normal(0.005, 0.03, size=60), index=index, name="net_ret")

    result = timing_gain_bootstrap(series, series.copy(), cfg)

    assert sorted(result) == ["diff", "p05", "p95", "p_one_sided"]
    assert result["diff"] == pytest.approx(0.0, abs=1e-12)
    assert result["p05"] == pytest.approx(0.0, abs=1e-12)
    assert result["p95"] == pytest.approx(0.0, abs=1e-12)
    assert result["p_one_sided"] == 1.0


def test_timing_gain_reproducible():
    cfg = _cfg(bootstrap_n_replications=50)
    rng = np.random.default_rng(cfg.run_seed)
    index = _months("2005-01-31", 60)
    static = pd.Series(rng.normal(0.004, 0.03, size=60), index=index)
    timed = pd.Series(rng.normal(0.006, 0.03, size=60), index=index)

    first = timing_gain_bootstrap(static, timed, cfg)
    second = timing_gain_bootstrap(static, timed, cfg)

    assert first == second


def test_timing_gain_is_signed_timed_minus_static():
    """A timed series that simply pays more has a positive diff and a low p_one_sided."""
    cfg = _cfg(bootstrap_n_replications=100)
    rng = np.random.default_rng(cfg.run_seed)
    index = _months("2005-01-31", 120)
    base = rng.normal(0.003, 0.02, size=120)
    static = pd.Series(base, index=index)
    timed = pd.Series(base + 0.006, index=index)           # same shocks, a higher mean

    result = timing_gain_bootstrap(static, timed, cfg)

    assert result["diff"] > 0
    assert result["p05"] <= result["p95"]
    assert result["p_one_sided"] < 0.1


def test_fill_timing_gain_completes_every_row():
    cfg = _cfg(strategy_min_regime_obs=6, bootstrap_n_replications=25)
    frames, factors = _grid_frames(cfg)
    grid = run_timing_grid(frames, factors, cfg)

    filled = fill_timing_gain(grid, frames, factors, cfg)

    assert list(filled.columns) == list(grid.columns)
    assert filled[["diff_p05", "diff_p95", "p_one_sided"]].notna().all().all()
    assert (filled["diff_p05"] <= filled["diff_p95"]).all()
    assert ((filled["p_one_sided"] >= 0) & (filled["p_one_sided"] <= 1)).all()
    pd.testing.assert_frame_equal(
        filled[["eta", "lag", "cost_bp", "source", "sharpe_static", "sharpe_timed", "diff"]],
        grid[["eta", "lag", "cost_bp", "source", "sharpe_static", "sharpe_timed", "diff"]],
    )
