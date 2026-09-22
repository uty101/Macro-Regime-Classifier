"""Timed factor allocation: trailing conditional Sharpe, weights, backtest, grid and timing-gain bootstrap. Built in section 5.

Everything here is a function of one *joined* frame -- the decision date t on
the index, the hard label and ``assigned`` flag in force at t, and the factor
returns of month t+1 -- built by ``regime.conditional.join_next_return``. That
is the only place the timing convention is applied, and importing nothing else
from the pipeline is what keeps it that way.

The strategy never loads a label source. ``run.py`` section 5 loads each label
frame and passes it in as a plain ``labels`` argument, so this module cannot
know, and cannot ask, which source it is holding. Step 3.8's test enforces
that on the text of this file: the hindsight labelling belongs in the
comparison rows of the timing grid and nowhere near the code that decides a
weight at t, and naming it here would itself trip the test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime.config import Config
from regime.conditional import join_next_return

ANNUALISE = 12
SQRT_ANNUALISE = float(np.sqrt(ANNUALISE))


def annualised_sharpe(returns: np.ndarray, ddof: int) -> float:
    """``mean / std(ddof) * sqrt(12)``; NaN on fewer than two observations or a zero spread."""
    r = returns[~np.isnan(returns)]
    if len(r) < 2:
        return float("nan")
    sd = float(r.std(ddof=ddof))
    if sd <= 0:
        return float("nan")
    return float(r.mean()) / sd * SQRT_ANNUALISE


def _trailing_from_joined(joined: pd.DataFrame, t: pd.Timestamp, k, cfg: Config) -> tuple[pd.Series, int]:
    """The trailing conditional Sharpe of regime ``k`` as at ``t``, from an already-joined frame.

    ``weights`` walks every decision date, so the join is done once by the
    caller and this is the per-date part. The public entry point below does
    the join itself and delegates here.
    """
    universe = list(cfg.strategy_universe)
    prior = joined.loc[joined.index < t]                      # strictly before t: s < t, never s == t
    if k is None or (isinstance(k, float) and np.isnan(k)):
        return pd.Series(np.nan, index=universe, name=t), 0

    same = prior.loc[prior["assigned"].to_numpy(dtype=bool) & (prior["label"] == k)]
    n_k = int(len(same))
    values = [annualised_sharpe(same[f].to_numpy(dtype="float64"), cfg.features_ddof) for f in universe]
    series = pd.Series(values, index=universe, name=t, dtype="float64")
    series.index.name = "factor"
    return series, n_k


def trailing_conditional_sharpe(
    labels: pd.DataFrame, factors: pd.DataFrame, t: pd.Timestamp, cfg: Config
) -> tuple[pd.Series, int]:
    """``(S, n_k)`` at decision date ``t``: the Sharpe of each allocation factor inside regime ``label_t``.

    ``k = label_t``. The sample is every decision date ``s < t`` -- strictly
    before, so the return the decision at t is about to earn is not in the
    sample that decides it -- with ``assigned_s`` true and ``label_s == k``,
    each contributing its ``return_{s+1}``. The Sharpe is
    ``mean / std(ddof) * sqrt(12)`` per factor in ``cfg.strategy_universe``,
    NaN where the regime has fewer than two prior observations. ``n_k`` is the
    number of such ``s`` and is the count ``weights`` tests against
    ``cfg.strategy_min_regime_obs``.

    The one thing this function must never do is reach forward. Month t+1 is
    earned by the decision at t, month t+2 by the decision at t+1, and neither
    decision date is ``< t``, so neither return can enter -- which is what
    ``tests/test_strategy.py::test_trailing_sharpe_is_strictly_before_t``
    plants a 1e3 return to check.
    """
    joined = join_next_return(labels, factors, cfg)
    if t not in labels.index:
        raise KeyError(f"decision date {t} is not in the label index")
    label_t = labels.loc[t, "label"]
    return _trailing_from_joined(joined, pd.Timestamp(t), label_t, cfg)


FALLBACK_UNASSIGNED = "unassigned"
FALLBACK_THIN_REGIME = "thin_regime"
FALLBACK_NO_POSITIVE = "no_positive_sharpe"
FALLBACK_NONE = ""


def static_weight(cfg: Config) -> float:
    """``1 / len(universe)`` -- 0.2 over the five allocation factors, and every fallback's answer."""
    return 1.0 / len(cfg.strategy_universe)


def _weight_row(s: pd.Series, n_k: int, assigned: bool, eta: float, cfg: Config) -> tuple[np.ndarray, str]:
    """One date's weight vector and the name of the branch that produced it.

    Three fallbacks, all to the static vector: the date is unassigned, the
    regime has fewer than ``cfg.strategy_min_regime_obs`` prior observations,
    or no factor has a positive trailing Sharpe in it. Otherwise
    ``w_f = (1 - eta) / 5 + eta * S+_f / sum_g S+_g`` with ``S+ = max(S, 0)``
    and NaN read as 0.

    The blend is a tilt, not a bet: every weight keeps ``(1 - eta) / 5`` of the
    static vector however large the Sharpe that earned the tilt, so a single
    thin regime cannot move a weight to 1.
    """
    universe = list(cfg.strategy_universe)
    flat = np.full(len(universe), static_weight(cfg))
    if not assigned:
        return flat, FALLBACK_UNASSIGNED
    if n_k < cfg.strategy_min_regime_obs:
        return flat, FALLBACK_THIN_REGIME

    positive = np.nan_to_num(s.to_numpy(dtype="float64"), nan=0.0)
    positive = np.where(positive > 0, positive, 0.0)
    total = float(positive.sum())
    if total == 0.0:
        return flat, FALLBACK_NO_POSITIVE

    return (1.0 - eta) * flat + eta * positive / total, FALLBACK_NONE


def weights(labels: pd.DataFrame, factors: pd.DataFrame, eta: float, cfg: Config) -> pd.DataFrame:
    """The timed weight vector at every out-of-sample decision date. Rows sum to 1.

    Index: the decision dates of ``join_next_return`` -- from
    ``cfg.sample_first_window_end`` to the last date whose t+1 return exists.
    Columns: ``cfg.strategy_universe`` (Mkt-RF is not allocable, convention 8).

    Each row uses only decision dates strictly before its own, through
    ``trailing_conditional_sharpe``. The first row therefore has no history at
    all and falls back, as does every date in a regime the strategy has seen
    fewer than ``cfg.strategy_min_regime_obs`` times.
    """
    joined = join_next_return(labels, factors, cfg)
    universe = list(cfg.strategy_universe)

    rows, branches = [], []
    for t in joined.index:
        assigned = bool(joined.at[t, "assigned"])
        s, n_k = _trailing_from_joined(joined, t, joined.at[t, "label"], cfg)
        row, branch = _weight_row(s, n_k, assigned, eta, cfg)
        rows.append(row)
        branches.append(branch)

    frame = pd.DataFrame(rows, index=joined.index, columns=universe, dtype="float64")
    frame.index.name = "date"
    frame.attrs["fallback_branch"] = pd.Series(branches, index=joined.index, name="fallback_branch")
    return frame


def static_weights(index: pd.DatetimeIndex, cfg: Config) -> pd.DataFrame:
    """``1 / len(universe)`` on every date of ``index``: the comparator the timed book is judged against."""
    frame = pd.DataFrame(
        static_weight(cfg), index=index, columns=list(cfg.strategy_universe), dtype="float64"
    )
    frame.index.name = "date"
    return frame


BACKTEST_COLUMNS = ("gross_ret", "turnover", "cost", "net_ret")


def backtest(weights: pd.DataFrame, factors: pd.DataFrame, lag: int, cost_bp: float, cfg: Config) -> pd.DataFrame:
    """The net return series of a weight book. Indexed by the *earning* month, not the decision date.

    The weights dated t earn the factor returns of the row ``1 + lag``
    positions after t in the monthly factor index: month t+1 at lag 0, month
    t+2 at lag 1. The shift is positional, on an index already asserted to be
    a complete monthly month-end sequence by ``join_next_return``, so a
    missing factor month cannot quietly turn one lag into another.

    ``turnover_t = 0.5 * sum_f |w_{f,t} - w_{f,t-1}|`` with the book starting
    from the static vector, so the first row's turnover is the cost of
    departing from 1/5 and is zero for a static book. ``cost`` is
    ``cost_bp / 1e4 * turnover_t`` and is deducted in the month the weights
    earn, not in the month they are decided -- the trade happens at the open
    of the earning month.

    Decision dates whose earning month is beyond the factor data are dropped.
    Only sleeve turnover is charged; the internal turnover of each
    Fama-French factor is identical across the static and timed books and is
    not modelled (convention 9).
    """
    universe = list(cfg.strategy_universe)
    index = pd.DatetimeIndex(factors.index)
    step = 1 + int(lag)

    position = index.get_indexer(weights.index)
    if (position < 0).any():
        missing = weights.index[position < 0]
        raise KeyError(f"{len(missing)} decision dates are absent from the factor index, first {missing[0]}")
    earning_position = position + step
    keep = earning_position < len(index)

    decision_dates = weights.index[keep]
    earning_months = index[earning_position[keep]]

    w = weights.loc[decision_dates, universe]
    returns = factors.loc[earning_months, universe]

    gross = (w.to_numpy(dtype="float64") * returns.to_numpy(dtype="float64")).sum(axis=1)

    previous = weights.loc[decision_dates, universe].shift(1)
    previous.iloc[0] = static_weight(cfg)                       # the book starts from the static vector
    turnover = 0.5 * (w - previous).abs().sum(axis=1).to_numpy(dtype="float64")
    cost = cost_bp / 1e4 * turnover

    out = pd.DataFrame(
        {"gross_ret": gross, "turnover": turnover, "cost": cost, "net_ret": gross - cost},
        index=pd.DatetimeIndex(earning_months, name="date"),
    )
    out.attrs["decision_date"] = pd.Series(decision_dates, index=out.index, name="decision_date")
    return out


GRID_COLUMNS = (
    "eta", "lag", "cost_bp", "source", "sharpe_static", "sharpe_timed", "diff",
    "diff_p05", "diff_p95", "p_one_sided", "mean_turnover", "n_months",
)
WEIGHT_DEVIATION_COLUMNS = (
    "source", "eta", "n_months", "n_months_off_static", "share_off_static",
    "mean_abs_deviation", "max_abs_deviation",
)
DEVIATION_TOLERANCE = 1e-9


def timing_cells(label_frames: dict[str, pd.DataFrame], factors: pd.DataFrame, cfg: Config) -> dict:
    """Every cell of the grid, keyed ``(eta, lag, cost_bp, source)``.

    Each value is ``{"static": ..., "timed": ..., "weights": ...}`` where the
    two net-return series share one index of earning months -- the timed and
    static books are built on the same decision dates and backtested at the
    same lag, so the comparison is never between two different samples of
    months. The weight books are built once per (source, eta) and reused
    across the lag and cost axes, which is the only reason 72 cells are cheap.
    """
    books = {
        (source, eta): weights(frame, factors, eta, cfg)
        for source, frame in label_frames.items()
        for eta in cfg.strategy_eta_grid
    }

    cells = {}
    for eta in cfg.strategy_eta_grid:
        for lag in cfg.strategy_lag_grid:
            for cost_bp in cfg.strategy_cost_bp_grid:
                for source in cfg.strategy_label_sources:
                    book = books[(source, eta)]
                    timed = backtest(book, factors, lag, cost_bp, cfg)
                    static = backtest(static_weights(book.index, cfg), factors, lag, cost_bp, cfg)
                    if not timed.index.equals(static.index):
                        raise ValueError(f"timed and static earning months differ for {(eta, lag, cost_bp, source)}")
                    cells[(eta, lag, cost_bp, source)] = {"static": static, "timed": timed, "weights": book}
    return cells


def run_timing_grid(label_frames: dict[str, pd.DataFrame], factors: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """The 3 x 2 x 3 x 4 = 72-row timing grid, with the interval and p columns left NaN for step 5.5.

    ``sharpe = mean(net_ret) / std(net_ret, ddof) * sqrt(12)`` over the common
    earning months, ``diff = sharpe_timed - sharpe_static``, ``mean_turnover``
    of the timed book.

    This is a grid and is reported as one. The single cell any summary may
    lead with is fixed in ``config.toml`` before any of it was computed:
    ``strategy.headline_eta``, ``headline_lag``, ``headline_cost_bp`` and
    ``headline_source``.
    """
    rows = []
    for (eta, lag, cost_bp, source), cell in timing_cells(label_frames, factors, cfg).items():
        static, timed = cell["static"], cell["timed"]
        sharpe_static = annualised_sharpe(static["net_ret"].to_numpy(dtype="float64"), cfg.features_ddof)
        sharpe_timed = annualised_sharpe(timed["net_ret"].to_numpy(dtype="float64"), cfg.features_ddof)
        rows.append(
            (eta, lag, cost_bp, source, sharpe_static, sharpe_timed, sharpe_timed - sharpe_static,
             np.nan, np.nan, np.nan, float(timed["turnover"].mean()), int(len(timed)))
        )
    return pd.DataFrame(rows, columns=list(GRID_COLUMNS))


def weight_deviation_summary(
    label_frames: dict[str, pd.DataFrame], factors: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    """How far the timed book ever actually moves from 1/5, per source and eta.

    A timing grid whose ``diff`` is near zero has two possible explanations: the
    tilt was taken and did not pay, or the tilt was never taken. This table
    separates them. ``n_months_off_static`` counts the months whose weight
    vector differs from the static one by more than 1e-9 in any component;
    ``mean_abs_deviation`` and ``max_abs_deviation`` are over every weight of
    every month, static rows included.
    """
    flat = static_weight(cfg)
    rows = []
    for source, frame in label_frames.items():
        for eta in cfg.strategy_eta_grid:
            book = weights(frame, factors, eta, cfg)
            deviation = (book.to_numpy(dtype="float64") - flat)
            off = (np.abs(deviation) > DEVIATION_TOLERANCE).any(axis=1)
            rows.append(
                (source, eta, int(len(book)), int(off.sum()), float(off.mean()),
                 float(np.abs(deviation).mean()), float(np.abs(deviation).max()))
            )
    return pd.DataFrame(rows, columns=list(WEIGHT_DEVIATION_COLUMNS))


def largest_turnover_months(result: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """The ``n`` largest single-month turnover values of one backtest, with their earning months.

    Ties are broken by the earning month so the table is deterministic.
    """
    frame = result[["turnover"]].copy()
    frame.index.name = "date"
    return (
        frame.reset_index()
        .sort_values(["turnover", "date"], ascending=[False, True])
        .head(n)
        .reset_index(drop=True)
    )
