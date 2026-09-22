"""Conditional factor statistics by regime, block bootstrap intervals and the filtered/smoothed gap. Built in section 4.

Everything here is a function of one *joined* frame: the decision date t on the
index, the hard label and ``assigned`` flag in force at t, and the factor
returns of month t+1. That frame is built once by ``join_next_return`` and is
the only place the timing convention is applied, so no statistic below can
reach a return dated on or before its own decision date.

The out-of-sample window is fixed by PLAN.md's conventions block: decision
dates from ``sample.first_window_end`` (2004-12-31) inclusive to the last
decision date whose t+1 return exists. Labels before that date exist for the
rules and smoothed sources and are consumed by nothing, so the four sources
are compared on identical dates.
"""

from __future__ import annotations

import pandas as pd

from regime.config import Config


def assert_complete_monthly(index: pd.DatetimeIndex) -> None:
    """Raise unless ``index`` is an ascending, gapless run of month-ends.

    ``join_next_return`` reads the t+1 return with ``shift(-1)``, which is a
    row shift, not a calendar one. A missing month in the factor file would
    silently turn "the return of month t+1" into "the return of month t+2",
    so the assumption that makes the shift legal is checked rather than
    assumed.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"factor index must be a DatetimeIndex, got {type(index).__name__}")
    expected = pd.date_range(index[0], index[-1], freq="ME")
    if not index.equals(expected):
        missing = expected.difference(index)
        extra = index.difference(expected)
        raise ValueError(
            f"factor index is not a complete monthly month-end sequence: "
            f"{len(missing)} missing ({list(missing[:5])}), {len(extra)} unexpected ({list(extra[:5])})"
        )


def join_next_return(labels: pd.DataFrame, factors: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """The decision at t beside the factor returns of month t+1.

    ``labels`` is indexed by decision date with columns ``label, assigned``;
    ``factors`` is indexed by month-end with the ``cfg.strategy_factors``
    columns. The result is indexed by the decision date t and carries
    ``label``, ``assigned`` and one column per factor holding that factor's
    month t+1 return. Decision dates with no t+1 return are dropped (the last
    row of the factor file), as are decision dates absent from the factor
    file. The frame is restricted to t >= ``cfg.sample_first_window_end``.
    """
    columns = list(cfg.strategy_factors)
    factors = factors[columns]
    assert_complete_monthly(pd.DatetimeIndex(factors.index))

    next_return = factors.shift(-1).iloc[:-1]          # the last month-end has no t+1 return
    joined = labels[["label", "assigned"]].join(next_return, how="inner")
    joined = joined.loc[joined.index >= pd.Timestamp(cfg.sample_first_window_end)]
    joined.index.name = "date"
    return joined

