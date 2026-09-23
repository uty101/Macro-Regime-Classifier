"""Fragility of the headline statistics: leave-one-month-out and trimmed differences. Built in step 6.7.

Section 5 found that the headline timing ``diff`` changes sign when one month
of 257 is removed, and section 4's one pairwise state difference that excludes
zero leans on a single month. Nothing else in ``PLAN.md`` catches that: step
6.5 varies the bootstrap's block size, which asks how much serial dependence
the interval carries, not how much of the point estimate one month is
carrying. This module asks the second question.

Neither statistic here is a result. Both are diagnostics *about* the headline
cell of the main run and about the one section 4 difference that excludes
zero, and neither replaces either. A statistic that survives leaving out its
worst month is not thereby significant; one that does not survive was never
worth reporting on its own.

Every function takes already-aligned net-return series or an already-joined
frame, so the timing convention is applied once, upstream, by
``regime.conditional.join_next_return``, and nothing here can reach a return
dated on or before its own decision date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime.config import Config
from regime.conditional import join_next_return, states_of
from regime.strategy import annualised_sharpe

LOO_COLUMNS = ("dropped_month", "diff", "sign_flipped")
FRAGILITY_COLUMNS = (
    "statistic", "full_value", "loo_min", "loo_max",
    "n_sign_flips", "share_sign_flips", "trimmed_value", "n_months",
)
DEFAULT_TRIM = 0.05


def _aligned(timed: pd.Series, static: pd.Series) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """The two net-return series on one index of earning months, or a ``ValueError``.

    An inner join is not enough: the point of the pairing is that each month
    contributes a (timed, static) pair from the *same* month, so a month
    present in one series and not the other is a bug in the caller, not a row
    to drop quietly.
    """
    frame = pd.concat([static.rename("static"), timed.rename("timed")], axis=1, join="outer")
    if frame.isna().any().any():
        raise ValueError("the timed and static series do not cover identical earning months")
    return (
        frame["timed"].to_numpy(dtype="float64"),
        frame["static"].to_numpy(dtype="float64"),
        pd.DatetimeIndex(frame.index, name="date"),
    )


def sharpe_difference(timed: np.ndarray, static: np.ndarray, cfg: Config) -> float:
    """``sharpe(timed) − sharpe(static)`` on two aligned arrays, the grid's own arithmetic."""
    return annualised_sharpe(timed, cfg.features_ddof) - annualised_sharpe(static, cfg.features_ddof)


def leave_one_month_out(timed: pd.Series, static: pd.Series, cfg: Config) -> pd.DataFrame:
    """One row per earning month: the headline ``diff`` recomputed with that month dropped.

    The month is dropped from **both** series, so every recomputation is still
    a like-for-like comparison of the two books over one common sample. Columns
    ``dropped_month``, ``diff`` and ``sign_flipped``, where ``sign_flipped`` is
    true when the recomputed difference lies strictly on the other side of zero
    from the difference over all months.

    A single sign flip is not a defect: it says the statistic is small relative
    to the influence of one month, which is what a null looks like. What the
    table is for is saying *how many* months can do it, and which.
    """
    timed_arr, static_arr, index = _aligned(timed, static)
    full = sharpe_difference(timed_arr, static_arr, cfg)

    rows = []
    for position, month in enumerate(index):
        keep = np.ones(len(index), dtype=bool)
        keep[position] = False
        value = sharpe_difference(timed_arr[keep], static_arr[keep], cfg)
        rows.append((month, value, bool(value * full < 0)))
    return pd.DataFrame(rows, columns=list(LOO_COLUMNS))


def _trim_count(n: int, trim: float) -> int:
    """How many months a ``trim`` fraction removes: ``floor(trim * n)``.

    Floor rather than round, so "the 5% of months with the largest difference"
    never removes more than 5% of them. At n = 257 that is 12 months (4.67%),
    not 13 (5.06%). Fixed here because the instruction fixes the fraction and
    not the rounding.
    """
    if not 0.0 <= trim < 1.0:
        raise ValueError(f"trim must be in [0, 1), got {trim}")
    return int(np.floor(trim * n))


def _drop_largest(values: np.ndarray, index: pd.DatetimeIndex, count: int) -> np.ndarray:
    """A boolean keep-mask removing the ``count`` largest ``values``, ties broken by earlier month.

    ``np.lexsort`` orders by the month first and by the magnitude second, so
    two months of identical influence are always dropped in calendar order and
    the answer does not depend on how the index happened to be built.
    """
    if count <= 0:
        return np.ones(len(values), dtype=bool)
    order = np.lexsort((index.to_numpy(), -values))
    keep = np.ones(len(values), dtype=bool)
    keep[order[:count]] = False
    return keep


def trimmed_diff(timed: pd.Series, static: pd.Series, trim: float = DEFAULT_TRIM,
                 cfg: Config | None = None) -> float:
    """The ``diff`` after removing the ``trim`` share of months with the largest |timed − static|.

    The months are removed from **both** series so they stay aligned and the
    two Sharpes are still computed over one common sample. ``cfg`` supplies
    only ``features_ddof``; it is optional so the function can be called with
    the plan's three-argument signature.

    This is the blunter companion to ``leave_one_month_out``: instead of asking
    what any one month does, it asks what is left when the handful of months
    that dominate the difference are gone.
    """
    ddof = 1 if cfg is None else cfg.features_ddof
    timed_arr, static_arr, index = _aligned(timed, static)
    influence = np.abs(timed_arr - static_arr)
    keep = _drop_largest(influence, index, _trim_count(len(index), trim))
    return (
        annualised_sharpe(timed_arr[keep], ddof) - annualised_sharpe(static_arr[keep], ddof)
    )


def trimmed_count(timed: pd.Series, static: pd.Series, trim: float = DEFAULT_TRIM) -> int:
    """How many months ``trimmed_diff`` removes from each series. Both get the same count."""
    _timed, _static, index = _aligned(timed, static)
    return _trim_count(len(index), trim)


def pairwise_months(labels: pd.DataFrame, factors: pd.DataFrame, factor: str,
                    state_a: int, state_b: int, cfg: Config) -> pd.DataFrame:
    """The assigned out-of-sample months in either state, with that factor's t+1 return.

    Columns ``state`` and ``ret``, indexed by the decision date. The same
    selection ``conditional_stats`` makes — assigned rows only, on
    ``join_next_return``'s window — restricted to the two states of the pair.
    """
    joined = join_next_return(labels, factors, cfg)
    present = states_of(joined)
    for state in (state_a, state_b):
        if state not in present:
            raise KeyError(f"state {state} is not among the assigned states {present}")
    use = joined.loc[joined["assigned"] & joined["label"].isin([state_a, state_b])]
    out = pd.DataFrame({"state": use["label"].astype("int64"), "ret": use[factor].astype("float64")})
    out.index.name = "date"
    return out.dropna()


def _pair_difference(state: np.ndarray, ret: np.ndarray, state_a: int, state_b: int,
                     cfg: Config) -> float:
    """``sharpe(state_b) − sharpe(state_a)``: the sign convention of ``conditional_differences``."""
    return (
        annualised_sharpe(ret[state == state_b], cfg.features_ddof)
        - annualised_sharpe(ret[state == state_a], cfg.features_ddof)
    )


def leave_one_month_out_pairwise(labels: pd.DataFrame, factors: pd.DataFrame, factor: str,
                                 state_a: int, state_b: int, cfg: Config) -> pd.DataFrame:
    """The same two statistics for a pairwise state difference of the conditional Sharpe.

    One row per month belonging to either state — the months the difference is
    computed from — carrying the difference recomputed with that month
    dropped, and whether that flips its sign.

    The trimmed version is the one place this cannot copy the headline's
    definition. For the headline there is a timed and a static return in the
    *same* month and "the largest timed-minus-static difference" is a per-month
    quantity. Here the two states occupy disjoint months and no such pairing
    exists, so the influence of a month is taken to be how far its removal
    moves the statistic, ``loo_diff − full_diff`` — which is what the
    leave-one-out column already measures. ``trimmed_pairwise_diff`` drops the
    5% of months with the largest absolute influence, from whichever state they
    fall in.
    """
    months = pairwise_months(labels, factors, factor, state_a, state_b, cfg)
    state = months["state"].to_numpy(dtype="int64")
    ret = months["ret"].to_numpy(dtype="float64")
    full = _pair_difference(state, ret, state_a, state_b, cfg)

    rows = []
    for position, month in enumerate(months.index):
        keep = np.ones(len(months), dtype=bool)
        keep[position] = False
        value = _pair_difference(state[keep], ret[keep], state_a, state_b, cfg)
        rows.append((month, value, bool(value * full < 0)))
    return pd.DataFrame(rows, columns=list(LOO_COLUMNS))


def trimmed_pairwise_diff(labels: pd.DataFrame, factors: pd.DataFrame, factor: str,
                          state_a: int, state_b: int, cfg: Config,
                          trim: float = DEFAULT_TRIM) -> float:
    """The pairwise difference after dropping the ``trim`` share of most influential months.

    Influence is ``|loo_diff − full_diff|``; see ``leave_one_month_out_pairwise``
    for why it is not the headline's definition.
    """
    months = pairwise_months(labels, factors, factor, state_a, state_b, cfg)
    state = months["state"].to_numpy(dtype="int64")
    ret = months["ret"].to_numpy(dtype="float64")
    full = _pair_difference(state, ret, state_a, state_b, cfg)

    loo = leave_one_month_out_pairwise(labels, factors, factor, state_a, state_b, cfg)
    influence = np.abs(loo["diff"].to_numpy(dtype="float64") - full)
    keep = _drop_largest(influence, pd.DatetimeIndex(months.index), _trim_count(len(months), trim))
    return _pair_difference(state[keep], ret[keep], state_a, state_b, cfg)


def fragility_row(statistic: str, full_value: float, loo: pd.DataFrame, trimmed_value: float) -> dict:
    """One row of ``fragility_summary.csv`` from a statistic's full value, its LOO table and its trim."""
    values = loo["diff"].to_numpy(dtype="float64")
    flips = int(loo["sign_flipped"].sum())
    return {
        "statistic": statistic,
        "full_value": float(full_value),
        "loo_min": float(values.min()),
        "loo_max": float(values.max()),
        "n_sign_flips": flips,
        "share_sign_flips": float(flips) / len(loo),
        "trimmed_value": float(trimmed_value),
        "n_months": int(len(loo)),
    }


def most_influential(loo: pd.DataFrame, full_value: float, n: int = 10) -> pd.DataFrame:
    """The ``n`` months whose removal moves a statistic most, with the size of the move.

    Sorted by ``abs_move`` descending and by ``dropped_month`` ascending, so
    the table is deterministic under ties.
    """
    frame = loo.copy()
    frame["move"] = frame["diff"] - full_value
    frame["abs_move"] = frame["move"].abs()
    return (
        frame.sort_values(["abs_move", "dropped_month"], ascending=[False, True])
        .head(n)
        .reset_index(drop=True)
    )
