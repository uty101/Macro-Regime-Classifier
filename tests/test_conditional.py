"""Section 4 — the join to t+1, conditional statistics, the block bootstrap and the gap."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from regime.conditional import join_next_return
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
