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


CI_COLUMNS = (
    "factor", "state", "n", "ann_mean", "ann_std", "sharpe", "sharpe_p05", "sharpe_p95", "excludes_zero",
)
DIFF_COLUMNS = (
    "factor", "state_a", "state_b", "sharpe_diff", "diff_p05", "diff_p95", "excludes_zero", "n_a", "n_b",
)
UNCONDITIONAL_COLUMNS = ("factor", "n", "ann_mean", "ann_std", "sharpe", "sharpe_p05", "sharpe_p95")
NAN_REPLICATION_COLUMNS = ("source", "factor", "state", "n_replications", "n_nan", "share_nan")


def excludes_zero(p_low, p_high):
    """True where the whole interval lies on one side of zero. NaN endpoints never exclude."""
    low, high = np.asarray(p_low, dtype="float64"), np.asarray(p_high, dtype="float64")
    return ((low > 0) | (high < 0)) & np.isfinite(low) & np.isfinite(high)


def _sharpe_by_state(label: np.ndarray, returns: np.ndarray, states: Sequence[int], ddof: int) -> np.ndarray:
    """(n_states, n_factors) annualised Sharpes; NaN for a state with fewer than 2 rows.

    A state with fewer than two observations has no sample standard deviation,
    so its Sharpe is NaN *for that replication* rather than zero or dropped.
    The percentiles are taken with ``np.nanquantile`` and the number of NaN
    replications per cell is reported alongside.
    """
    out = np.full((len(states), returns.shape[1]), np.nan)
    for i, state in enumerate(states):
        rows = returns[label == state]
        if len(rows) < 2:
            continue
        mean, sd = rows.mean(axis=0), rows.std(axis=0, ddof=ddof)
        safe = np.where(sd > 0, sd, 1.0)
        out[i] = np.where(sd > 0, mean / safe * SQRT_ANNUALISE, np.nan)
    return out


def state_pairs(states: Sequence[int]) -> list[tuple[int, int]]:
    """Every (a, b) with a < b, ascending: 3 pairs at K = 3, 6 at K = 4."""
    return [(a, b) for i, a in enumerate(states) for b in states[i + 1 :]]


def _replication_matrices(joined: pd.DataFrame, states: Sequence[int], cfg: Config):
    """Sharpe and pairwise-difference arrays over ``cfg.bootstrap_n_replications`` draws.

    The whole joined frame is resampled jointly -- one row is one (label,
    assigned, six returns) tuple -- so a label and the returns it is judged
    against can never come from different months, and the unassigned rows stay
    in the series rather than being deleted before the blocks are drawn.
    Within each draw the assigned rows are selected exactly as
    ``conditional_stats`` selects them.

    Returns ``(sharpe, diff)`` of shapes (R, n_states, n_factors) and
    (R, n_pairs, n_factors). Each difference is formed inside its own
    replication from that replication's Sharpes, never from two independently
    resampled marginals -- which is the whole reason the two tables share one
    bootstrap.
    """
    from arch.bootstrap import StationaryBootstrap

    factor_names = list(cfg.strategy_factors)
    arr = np.column_stack(
        [
            joined["label"].to_numpy(dtype="float64"),
            joined["assigned"].to_numpy(dtype="float64"),
            joined[factor_names].to_numpy(dtype="float64"),
        ]
    )
    reps = cfg.bootstrap_n_replications
    pairs = state_pairs(states)
    sharpe = np.full((reps, len(states), len(factor_names)), np.nan)
    diff = np.full((reps, len(pairs), len(factor_names)), np.nan)
    position = {state: i for i, state in enumerate(states)}

    bootstrap = StationaryBootstrap(cfg.bootstrap_block_size, arr, seed=cfg.run_seed)
    for r, ((draw,), _) in enumerate(bootstrap.bootstrap(reps)):
        assigned = draw[draw[:, 1] > 0.5]
        s = _sharpe_by_state(assigned[:, 0], assigned[:, 2:], states, cfg.features_ddof)
        sharpe[r] = s
        for j, (a, b) in enumerate(pairs):
            diff[r, j] = s[position[b]] - s[position[a]]
    return sharpe, diff


def bootstrap_conditional(labels: pd.DataFrame, factors: pd.DataFrame, cfg: Config, source: str = ""):
    """``(conditional_stats_with_ci, conditional_differences, nan_replication_counts)``.

    One bootstrap, three tables: the per-cell intervals of step 4.3, the
    pairwise state differences, and the count of replications in which a
    cell's Sharpe was NaN.
    """
    joined = join_next_return(labels, factors, cfg)
    states = states_of(joined)
    observed = conditional_stats_from_joined(joined, cfg)
    sharpe, diff = _replication_matrices(joined, states, cfg)

    factor_names = list(cfg.strategy_factors)
    low = np.nanquantile(sharpe, cfg.bootstrap_p_low, axis=0)
    high = np.nanquantile(sharpe, cfg.bootstrap_p_high, axis=0)
    n_nan = np.isnan(sharpe).sum(axis=0)

    lookup = {(row.factor, row.state): row for row in observed.itertuples(index=False)}
    stats_rows, nan_rows = [], []
    for fi, factor in enumerate(factor_names):
        for si, state in enumerate(states):
            row = lookup[(factor, state)]
            stats_rows.append(
                (factor, state, row.n, row.ann_mean, row.ann_std, row.sharpe,
                 low[si, fi], high[si, fi], bool(excludes_zero(low[si, fi], high[si, fi])))
            )
            nan_rows.append(
                (source, factor, state, cfg.bootstrap_n_replications,
                 int(n_nan[si, fi]), float(n_nan[si, fi]) / cfg.bootstrap_n_replications)
            )
    stats = pd.DataFrame(stats_rows, columns=list(CI_COLUMNS))

    diff_low = np.nanquantile(diff, cfg.bootstrap_p_low, axis=0)
    diff_high = np.nanquantile(diff, cfg.bootstrap_p_high, axis=0)
    diff_rows = []
    for fi, factor in enumerate(factor_names):
        for pi, (a, b) in enumerate(state_pairs(states)):
            row_a, row_b = lookup[(factor, a)], lookup[(factor, b)]
            diff_rows.append(
                (factor, a, b, row_b.sharpe - row_a.sharpe, diff_low[pi, fi], diff_high[pi, fi],
                 bool(excludes_zero(diff_low[pi, fi], diff_high[pi, fi])), row_a.n, row_b.n)
            )
    differences = pd.DataFrame(diff_rows, columns=list(DIFF_COLUMNS))

    return stats, differences, pd.DataFrame(nan_rows, columns=list(NAN_REPLICATION_COLUMNS))


def block_bootstrap_ci(labels: pd.DataFrame, factors: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """``conditional_stats`` with stationary-bootstrap percentiles and ``excludes_zero``."""
    return bootstrap_conditional(labels, factors, cfg)[0]


def conditional_differences(labels: pd.DataFrame, factors: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Pairwise state differences of the conditional Sharpe, from the same bootstrap."""
    return bootstrap_conditional(labels, factors, cfg)[1]


def unconditional_stats(labels: pd.DataFrame, factors: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Pooled factor statistics over the same out-of-sample dates, all months included.

    No state conditioning and no ``assigned`` filter: this is the number every
    conditional Sharpe has to be read against, on exactly the dates the
    conditional tables use. ``labels`` is taken only for its index.
    """
    from arch.bootstrap import StationaryBootstrap

    joined = join_next_return(labels, factors, cfg)
    factor_names = list(cfg.strategy_factors)
    returns = joined[factor_names]

    reps = cfg.bootstrap_n_replications
    draws = np.full((reps, len(factor_names)), np.nan)
    arr = returns.to_numpy(dtype="float64")
    bootstrap = StationaryBootstrap(cfg.bootstrap_block_size, arr, seed=cfg.run_seed)
    for r, ((draw,), _) in enumerate(bootstrap.bootstrap(reps)):
        mean, sd = draw.mean(axis=0), draw.std(axis=0, ddof=cfg.features_ddof)
        safe = np.where(sd > 0, sd, 1.0)
        draws[r] = np.where(sd > 0, mean / safe * SQRT_ANNUALISE, np.nan)

    low = np.nanquantile(draws, cfg.bootstrap_p_low, axis=0)
    high = np.nanquantile(draws, cfg.bootstrap_p_high, axis=0)
    rows = []
    for fi, factor in enumerate(factor_names):
        n, ann_mean, ann_std, sharpe = _moments(returns[factor], cfg.features_ddof)
        rows.append((factor, n, ann_mean, ann_std, sharpe, low[fi], high[fi]))
    return pd.DataFrame(rows, columns=list(UNCONDITIONAL_COLUMNS))


GAP_COLUMNS = ("factor", "state", "gap", "gap_p05", "gap_p95")


def join_both(filt: pd.DataFrame, smooth: pd.DataFrame, factors: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """One frame carrying both label pairs and the t+1 returns, on the common out-of-sample dates.

    ``label_f, assigned_f, label_s, assigned_s`` and one column per factor.
    Both sources are joined to the *same* return columns, so the gap below is
    a difference between two labellings of one return series and nothing else.
    """
    joined_f = join_next_return(filt, factors, cfg)
    joined_s = join_next_return(smooth, factors, cfg)
    common = joined_f.index.intersection(joined_s.index)

    frame = pd.DataFrame(
        {
            "label_f": joined_f.loc[common, "label"],
            "assigned_f": joined_f.loc[common, "assigned"],
            "label_s": joined_s.loc[common, "label"],
            "assigned_s": joined_s.loc[common, "assigned"],
        },
        index=common,
    )
    for factor in cfg.strategy_factors:
        frame[factor] = joined_f.loc[common, factor]
    frame.index.name = "date"
    return frame


def _gap_states(frame: pd.DataFrame) -> list[int]:
    """The states either labelling assigns, ascending.

    Smoothed and filtered state numbers already agree through chained
    anchoring (convention 16), so state k means the same regime in both and
    nothing is relabelled here.
    """
    filtered = frame.loc[frame["assigned_f"], "label_f"].dropna()
    smoothed = frame.loc[frame["assigned_s"], "label_s"].dropna()
    return sorted({int(k) for k in filtered.unique()} | {int(k) for k in smoothed.unique()})


def _gap_matrix(frame_arr: np.ndarray, states: Sequence[int], n_factors: int, ddof: int) -> np.ndarray:
    """(n_states, n_factors) smoothed-minus-filtered Sharpe for one sample.

    ``frame_arr`` columns are label_f, assigned_f, label_s, assigned_s, then
    the returns.
    """
    returns = frame_arr[:, 4:]
    keep_f = frame_arr[:, 1] > 0.5
    keep_s = frame_arr[:, 3] > 0.5
    sharpe_f = _sharpe_by_state(frame_arr[keep_f, 0], returns[keep_f], states, ddof)
    sharpe_s = _sharpe_by_state(frame_arr[keep_s, 2], returns[keep_s], states, ddof)
    return sharpe_s - sharpe_f


def filtered_smoothed_gap(
    filt: pd.DataFrame, smooth: pd.DataFrame, factors: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    """How much of each conditional Sharpe is hindsight: smoothed minus filtered.

    The filtered and smoothed labels are resampled in the *same* draw -- one
    row of the bootstrap array carries both labellings and the returns -- so
    each replication's gap is a like-for-like comparison on one resampled
    history. Resampling the two separately would add a difference between two
    samples to the difference between two labellings.

    Columns ``factor, state, gap, gap_p05, gap_p95``; ``gap`` is the observed
    difference, the percentiles are ``cfg.bootstrap_p_low`` and
    ``cfg.bootstrap_p_high`` of the replications.
    """
    from arch.bootstrap import StationaryBootstrap

    frame = join_both(filt, smooth, factors, cfg)
    states = _gap_states(frame)
    factor_names = list(cfg.strategy_factors)

    arr = np.column_stack(
        [
            frame["label_f"].to_numpy(dtype="float64"),
            frame["assigned_f"].to_numpy(dtype="float64"),
            frame["label_s"].to_numpy(dtype="float64"),
            frame["assigned_s"].to_numpy(dtype="float64"),
            frame[factor_names].to_numpy(dtype="float64"),
        ]
    )
    observed = _gap_matrix(arr, states, len(factor_names), cfg.features_ddof)

    reps = cfg.bootstrap_n_replications
    draws = np.full((reps, len(states), len(factor_names)), np.nan)
    bootstrap = StationaryBootstrap(cfg.bootstrap_block_size, arr, seed=cfg.run_seed)
    for r, ((draw,), _) in enumerate(bootstrap.bootstrap(reps)):
        draws[r] = _gap_matrix(draw, states, len(factor_names), cfg.features_ddof)

    low = np.nanquantile(draws, cfg.bootstrap_p_low, axis=0)
    high = np.nanquantile(draws, cfg.bootstrap_p_high, axis=0)
    rows = []
    for fi, factor in enumerate(factor_names):
        for si, state in enumerate(states):
            rows.append((factor, state, observed[si, fi], low[si, fi], high[si, fi]))
    return pd.DataFrame(rows, columns=list(GAP_COLUMNS))
