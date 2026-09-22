"""Section 4 — the join to t+1, conditional statistics, the block bootstrap and the gap."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from regime.conditional import (
    conditional_stats_from_joined,
    join_next_return,
    run_started_on_refit,
    unassigned_dates,
)
from regime.config import load_config


def _months(start: str, n: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="ME", name="date")


def test_join_alignment_on_four_rows():
    """The decision at t earns month t+1, and the last decision date is dropped.

    Four decision dates 2020-01-31 .. 2020-04-30 against five factor months
    2020-01 .. 2020-05, every return distinct, so a one-row misalignment
    cannot pass. The 2020-05-31 decision date has no t+1 return and is absent.
    """
    cfg = dataclasses.replace(load_config(), sample_first_window_end="2020-01-31")
    factor_index = _months("2020-01-31", 5)
    factors = pd.DataFrame(
        {f: np.arange(5, dtype=float) + offset for offset, f in enumerate(cfg.strategy_factors)},
        index=factor_index,
    )
    labels = pd.DataFrame(
        {"label": [0, 1, 2, 0, 1], "assigned": [True, True, True, True, True]},
        index=factor_index,
    )

    joined = join_next_return(labels, factors, cfg)

    assert list(joined.index) == list(factor_index[:4])
    assert pd.Timestamp("2020-05-31") not in joined.index
    assert list(joined.columns) == ["label", "assigned"] + list(cfg.strategy_factors)
    # the return at decision date 2020-01-31 is the 2020-02 return, and so on
    for i, t in enumerate(factor_index[:4]):
        for f in cfg.strategy_factors:
            assert joined.loc[t, f] == factors.loc[factor_index[i + 1], f]


def test_join_drops_dates_before_first_window_end():
    cfg = dataclasses.replace(load_config(), sample_first_window_end="2020-03-31")
    factor_index = _months("2020-01-31", 5)
    factors = pd.DataFrame({f: np.arange(5, dtype=float) for f in cfg.strategy_factors}, index=factor_index)
    labels = pd.DataFrame({"label": [0] * 5, "assigned": [True] * 5}, index=factor_index)

    joined = join_next_return(labels, factors, cfg)

    assert list(joined.index) == [pd.Timestamp("2020-03-31"), pd.Timestamp("2020-04-30")]


def test_join_rejects_a_gap_in_the_factor_index():
    """A missing factor month would turn the row shift into a t+2 return."""
    cfg = load_config()
    index = _months("2020-01-31", 5).delete(2)
    factors = pd.DataFrame({f: np.arange(4, dtype=float) for f in cfg.strategy_factors}, index=index)
    labels = pd.DataFrame({"label": [0] * 4, "assigned": [True] * 4}, index=index)

    with pytest.raises(ValueError, match="complete monthly month-end sequence"):
        join_next_return(labels, factors, cfg)


def _joined(labels, returns, factors):
    """A joined frame built by hand: ``label``, ``assigned`` and one column per factor."""
    index = _months("2005-01-31", len(labels))
    frame = pd.DataFrame({"label": labels, "assigned": [True] * len(labels)}, index=index)
    for f in factors:
        frame[f] = returns[f]
    return frame


def test_conditional_stats_by_hand():
    """Literal expected n, ann_mean, ann_std and sharpe for two states and two factors."""
    cfg = dataclasses.replace(load_config(), strategy_factors=("Mkt-RF", "SMB"))
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    returns = {
        "Mkt-RF": [0.01, 0.03, 0.02, 0.02, -0.01, 0.01, 0.00, 0.00],
        "SMB": [0.00, 0.02, -0.02, 0.04, 0.05, 0.05, 0.05, 0.01],
    }
    joined = _joined(labels, returns, cfg.strategy_factors)

    stats = conditional_stats_from_joined(joined, cfg)

    assert list(stats.columns) == ["factor", "state", "n", "ann_mean", "ann_std", "sharpe"]
    assert list(stats["factor"]) == ["Mkt-RF", "Mkt-RF", "SMB", "SMB"]
    assert list(stats["state"]) == [0, 1, 0, 1]
    assert list(stats["n"]) == [4, 4, 4, 4]

    sqrt12 = np.sqrt(12.0)
    for row in stats.itertuples(index=False):
        sample = np.array(returns[row.factor][:4] if row.state == 0 else returns[row.factor][4:])
        assert row.ann_mean == pytest.approx(sample.mean() * 12, abs=1e-12)
        assert row.ann_std == pytest.approx(sample.std(ddof=1) * sqrt12, abs=1e-12)
        assert row.sharpe == pytest.approx(sample.mean() / sample.std(ddof=1) * sqrt12, abs=1e-12)

    # the two literal cells, computed independently of the loop above
    mkt0 = stats[(stats.factor == "Mkt-RF") & (stats.state == 0)].iloc[0]
    assert mkt0.ann_mean == pytest.approx(0.02 * 12, abs=1e-12)
    assert mkt0.ann_std == pytest.approx(np.sqrt(((np.array([0.01, 0.03, 0.02, 0.02]) - 0.02) ** 2).sum() / 3) * sqrt12, abs=1e-12)


def test_unassigned_rows_are_excluded_and_counted():
    cfg = dataclasses.replace(load_config(), strategy_factors=("Mkt-RF",))
    index = _months("2005-01-31", 6)
    joined = pd.DataFrame(
        {
            "label": [0, 0, 0, 1, 1, 1],
            "assigned": [True, True, False, True, True, False],
            "Mkt-RF": [0.01, 0.03, 99.0, -0.01, 0.01, -99.0],
        },
        index=index,
    )

    stats = conditional_stats_from_joined(joined, cfg)

    assert list(stats["n"]) == [2, 2]
    assert stats.loc[stats.state == 0, "ann_mean"].iloc[0] == pytest.approx(0.02 * 12, abs=1e-12)
    assert stats.loc[stats.state == 1, "ann_mean"].iloc[0] == pytest.approx(0.0, abs=1e-12)

    labels = joined[["label", "assigned"]]
    factors = pd.DataFrame({"Mkt-RF": [0.0] * 7}, index=_months("2005-01-31", 7))
    assert list(unassigned_dates(labels, factors, dataclasses.replace(cfg, sample_first_window_end="2005-01-31"))) == [
        pd.Timestamp("2005-03-31"),
        pd.Timestamp("2005-06-30"),
    ]


def test_run_started_on_refit_flag():
    """Eight dates, two refit dates, three runs; the flag is carried from each run's start."""
    index = _months("2005-01-31", 8)
    labels = pd.DataFrame({"label": [0, 0, 1, 1, 1, 2, 2, 2], "assigned": [True] * 8}, index=index)
    refits = [pd.Timestamp("2005-03-31"), pd.Timestamp("2005-07-31")]

    flag = run_started_on_refit(labels, refits)

    # run 1 starts 2005-01-31 (not a refit); run 2 starts 2005-03-31 (a refit);
    # run 3 starts 2005-06-30 (not a refit), so 2005-07-31 being a refit is irrelevant
    assert list(flag) == [False, False, True, True, True, False, False, False]
    assert list(flag.index) == list(index)
