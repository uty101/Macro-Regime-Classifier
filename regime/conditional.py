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

from collections.abc import Sequence

import numpy as np
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


ANNUALISE = 12
SQRT_ANNUALISE = float(np.sqrt(ANNUALISE))

STATS_COLUMNS = ("factor", "state", "n", "ann_mean", "ann_std", "sharpe")
REFIT_SPLIT_COLUMNS = ("factor", "state", "run_started_on_refit", "n", "ann_mean", "ann_std", "sharpe")


def states_of(joined: pd.DataFrame) -> list[int]:
    """The hard labels present among the assigned rows, ascending.

    Read off the data rather than from ``primary_K`` so the same function
    serves the four sources: ``rules`` has four states, the HMM and GMM have
    ``primary_K``.
    """
    return sorted(int(k) for k in joined.loc[joined["assigned"], "label"].dropna().unique())


def _moments(returns: pd.Series, ddof: int) -> tuple[int, float, float, float]:
    """``n``, annualised mean, annualised standard deviation and Sharpe of one return sample."""
    r = returns.dropna()
    n = int(len(r))
    if n < 2:
        return n, float(r.mean()) * ANNUALISE if n else float("nan"), float("nan"), float("nan")
    mean, sd = float(r.mean()), float(r.std(ddof=ddof))
    sharpe = mean / sd * SQRT_ANNUALISE if sd > 0 else float("nan")
    return n, mean * ANNUALISE, sd * SQRT_ANNUALISE, sharpe


def conditional_stats_from_joined(joined: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """``factor, state, n, ann_mean, ann_std, sharpe`` over the assigned rows of a joined frame."""
    use = joined.loc[joined["assigned"]]
    rows = []
    for factor in cfg.strategy_factors:
        for state in states_of(joined):
            n, ann_mean, ann_std, sharpe = _moments(use.loc[use["label"] == state, factor], cfg.features_ddof)
            rows.append((factor, state, n, ann_mean, ann_std, sharpe))
    return pd.DataFrame(rows, columns=list(STATS_COLUMNS))


def conditional_stats(labels: pd.DataFrame, factors: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Conditional factor statistics by hard label over the out-of-sample window.

    Unassigned rows (max probability at or below ``cfg.hmm_assigned_threshold``)
    are excluded; ``n`` is the number of assigned rows behind each cell and
    appears on every row. Annualisation is x12 for the mean and x sqrt(12) for
    the standard deviation and the Sharpe; the standard deviation uses
    ``cfg.features_ddof``.
    """
    return conditional_stats_from_joined(join_next_return(labels, factors, cfg), cfg)


def unassigned_dates(labels: pd.DataFrame, factors: pd.DataFrame, cfg: Config) -> pd.DatetimeIndex:
    """The out-of-sample decision dates a source leaves unassigned, for the review file."""
    joined = join_next_return(labels, factors, cfg)
    return pd.DatetimeIndex(joined.index[~joined["assigned"].to_numpy(dtype=bool)], name="date")


def run_started_on_refit(labels: pd.DataFrame, refits: Sequence[pd.Timestamp]) -> pd.Series:
    """True at every date whose label run began on a refit date.

    A run is a maximal block of consecutive equal labels in ``labels``; the
    first row of the frame starts a run. The flag is carried forward from each
    run's start, so it answers "was this regime first entered on the day the
    model was refitted?" rather than "is today a refit date?".

    This is the measurement the reviewer asked for in answer to Q1
    (``decisions/section_3_review.md``): 45% of filtered label changes land on
    a 31 December refit, and the cost of that is whatever separates the two
    halves of this split.
    """
    label = labels["label"]
    new_run = label.ne(label.shift())
    new_run.iloc[0] = True
    start = pd.Series(pd.NaT, index=label.index, dtype="datetime64[ns]")
    start.loc[new_run] = label.index[new_run.to_numpy(dtype=bool)]
    start = start.ffill()
    flag = start.isin(pd.DatetimeIndex(refits))
    flag.name = "run_started_on_refit"
    flag.index.name = "date"
    return flag


def conditional_stats_refit_split(
    labels: pd.DataFrame, factors: pd.DataFrame, refits: Sequence[pd.Timestamp], cfg: Config
) -> pd.DataFrame:
    """Conditional statistics by factor, state and ``run_started_on_refit``. Information only, no bootstrap."""
    joined = join_next_return(labels, factors, cfg)
    flag = run_started_on_refit(labels, refits).reindex(joined.index)
    use = joined.loc[joined["assigned"]]
    flagged = flag.reindex(use.index).astype(bool)

    rows = []
    for factor in cfg.strategy_factors:
        for state in states_of(joined):
            for value in (False, True):
                mask = (use["label"] == state) & (flagged == value)
                n, ann_mean, ann_std, sharpe = _moments(use.loc[mask, factor], cfg.features_ddof)
                rows.append((factor, state, value, n, ann_mean, ann_std, sharpe))
    return pd.DataFrame(rows, columns=list(REFIT_SPLIT_COLUMNS))
