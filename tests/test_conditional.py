"""Section 4 — the join to t+1, conditional statistics, the block bootstrap and the gap."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from regime.conditional import (
    add_excess_sharpe,
    block_bootstrap_ci,
    bootstrap_refit_split,
    bootstrap_conditional,
    conditional_stats,
    conditional_stats_from_joined,
    excludes_zero,
    filtered_smoothed_gap,
    state_pairs,
    unconditional_stats,
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


def _bootstrap_cfg(**kwargs):
    """The real config with a small replication count, so a test is seconds not minutes."""
    base = dataclasses.replace(load_config(), bootstrap_n_replications=100, strategy_factors=("Mkt-RF", "SMB"))
    return dataclasses.replace(base, **kwargs)


def _labels_and_factors(labels, returns, factor_names, first_window_end="2005-01-31"):
    """A label frame and a factor frame that ``join_next_return`` turns into the intended joined frame.

    The factor frame carries one extra month at the end, because the decision
    at the last labelled date earns the month after it.
    """
    n = len(labels)
    index = _months("2005-01-31", n)
    factor_index = _months("2005-01-31", n + 1)
    label_frame = pd.DataFrame({"label": labels, "assigned": [True] * n}, index=index)
    factors = pd.DataFrame(
        {f: [0.0] + list(returns[f]) for f in factor_names}, index=factor_index
    )
    return label_frame, factors, first_window_end


def test_excludes_zero_rule():
    cases = [(0.1, 0.4, True), (-0.4, -0.1, True), (-0.1, 0.4, False), (0.0, 0.4, False),
             (-0.4, 0.0, False), (np.nan, 0.4, False), (0.1, np.nan, False)]
    low = np.array([c[0] for c in cases])
    high = np.array([c[1] for c in cases])
    assert list(excludes_zero(low, high)) == [c[2] for c in cases]


def test_bootstrap_reproducible_with_config_seed():
    cfg = _bootstrap_cfg()
    rng = np.random.default_rng(cfg.run_seed)
    labels = list(rng.integers(0, 3, size=60))
    returns = {f: rng.normal(0.005, 0.03, size=60) for f in cfg.strategy_factors}
    label_frame, factors, fwe = _labels_and_factors(labels, returns, cfg.strategy_factors)
    cfg = dataclasses.replace(cfg, sample_first_window_end=fwe)

    first = block_bootstrap_ci(label_frame, factors, cfg)
    second = block_bootstrap_ci(label_frame, factors, cfg)

    pd.testing.assert_frame_equal(first, second)


def test_bootstrap_schema_and_n_unchanged():
    cfg = _bootstrap_cfg()
    rng = np.random.default_rng(cfg.run_seed)
    labels = list(rng.integers(0, 3, size=60))
    returns = {f: rng.normal(0.005, 0.03, size=60) for f in cfg.strategy_factors}
    label_frame, factors, fwe = _labels_and_factors(labels, returns, cfg.strategy_factors)
    cfg = dataclasses.replace(cfg, sample_first_window_end=fwe)

    with_ci = block_bootstrap_ci(label_frame, factors, cfg)
    plain = conditional_stats(label_frame, factors, cfg)

    assert list(with_ci.columns) == [
        "factor", "state", "n", "ann_mean", "ann_std", "sharpe", "sharpe_p05", "sharpe_p95", "excludes_zero",
    ]
    assert len(with_ci.columns) == 9
    pd.testing.assert_frame_equal(with_ci[list(plain.columns)], plain)


def test_pairwise_difference_zero_when_states_identical():
    """Two states whose return rows are identical differ by exactly 0, in every replication.

    ``bootstrap_block_size`` is set far above the sample length, so every
    stationary-bootstrap draw is a single circular block covering the whole
    series: a rotation. A rotation preserves each state's rows exactly, so if
    the two states hold the same returns their Sharpes coincide in every
    replication and both percentiles of the difference are 0. That is only
    true because the difference is formed inside each replication; had the two
    states been resampled independently the percentiles would straddle 0.
    """
    cfg = _bootstrap_cfg(bootstrap_block_size=10 ** 9, bootstrap_n_replications=50)
    values = [0.01, -0.02, 0.03, 0.005, -0.01, 0.02]
    labels, returns = [], {f: [] for f in cfg.strategy_factors}
    for v in values:                       # state 0 and state 1 get the same six returns
        for state in (0, 1):
            labels.append(state)
            for f in cfg.strategy_factors:
                returns[f].append(v)
    label_frame, factors, fwe = _labels_and_factors(labels, returns, cfg.strategy_factors)
    cfg = dataclasses.replace(cfg, sample_first_window_end=fwe)

    _stats, differences, _nan = bootstrap_conditional(label_frame, factors, cfg)

    assert list(differences.columns) == [
        "factor", "state_a", "state_b", "sharpe_diff", "diff_p05", "diff_p95", "excludes_zero", "n_a", "n_b",
    ]
    assert list(differences["state_a"]) == [0] * len(cfg.strategy_factors)
    assert list(differences["state_b"]) == [1] * len(cfg.strategy_factors)
    assert list(differences["n_a"]) == [6] * len(cfg.strategy_factors)
    assert list(differences["n_b"]) == [6] * len(cfg.strategy_factors)
    for row in differences.itertuples(index=False):
        assert row.sharpe_diff == pytest.approx(0.0, abs=1e-12)
        assert row.diff_p05 == pytest.approx(0.0, abs=1e-12)
        assert row.diff_p95 == pytest.approx(0.0, abs=1e-12)
        assert row.excludes_zero is False


def test_pair_count_is_three_at_k3_and_six_at_k4():
    assert len(state_pairs([0, 1, 2])) == 3
    assert len(state_pairs([0, 1, 2, 3])) == 6


def test_unconditional_stats_uses_every_month_and_the_same_dates():
    cfg = _bootstrap_cfg()
    rng = np.random.default_rng(cfg.run_seed)
    labels = list(rng.integers(0, 3, size=60))
    returns = {f: rng.normal(0.005, 0.03, size=60) for f in cfg.strategy_factors}
    label_frame, factors, fwe = _labels_and_factors(labels, returns, cfg.strategy_factors)
    cfg = dataclasses.replace(cfg, sample_first_window_end=fwe)
    # half the rows unassigned: the unconditional table must ignore the flag
    label_frame = label_frame.copy()
    label_frame.iloc[::2, label_frame.columns.get_loc("assigned")] = False

    table = unconditional_stats(label_frame, factors, cfg)

    assert list(table.columns) == ["factor", "n", "ann_mean", "ann_std", "sharpe", "sharpe_p05", "sharpe_p95"]
    assert list(table["n"]) == [60, 60]
    for row in table.itertuples(index=False):
        sample = np.array(returns[row.factor])
        assert row.ann_mean == pytest.approx(sample.mean() * 12, abs=1e-12)
        assert row.sharpe == pytest.approx(sample.mean() / sample.std(ddof=1) * np.sqrt(12), abs=1e-12)
        assert row.sharpe_p05 <= row.sharpe_p95


def test_gap_is_zero_when_labels_identical():
    """One labelling used twice has no hindsight in it, in every replication.

    The filtered and smoothed labels travel in the same bootstrap row, so when
    they are equal the two Sharpes are computed from the same draw and the gap
    is exactly 0 -- not merely centred on 0.
    """
    cfg = _bootstrap_cfg()
    rng = np.random.default_rng(cfg.run_seed)
    labels = list(rng.integers(0, 3, size=60))
    returns = {f: rng.normal(0.005, 0.03, size=60) for f in cfg.strategy_factors}
    label_frame, factors, fwe = _labels_and_factors(labels, returns, cfg.strategy_factors)
    cfg = dataclasses.replace(cfg, sample_first_window_end=fwe)

    gap = filtered_smoothed_gap(label_frame, label_frame.copy(), factors, cfg)

    assert list(gap.columns) == ["factor", "state", "gap", "gap_p05", "gap_p95"]
    assert len(gap) == len(cfg.strategy_factors) * 3
    for row in gap.itertuples(index=False):
        assert row.gap == pytest.approx(0.0, abs=1e-12)
        assert row.gap_p05 == pytest.approx(0.0, abs=1e-12)
        assert row.gap_p95 == pytest.approx(0.0, abs=1e-12)


def test_gap_sign_is_smoothed_minus_filtered():
    """A smoothed labelling that sorts the good months into state 1 gives a positive gap there.

    The filtered labelling alternates, so both its states hold two good and two
    bad months and both its Sharpes are 0. The smoothed labelling has the four
    good months in state 1 and the four bad ones in state 0. The gap must
    therefore be positive in state 1 and negative in state 0: hindsight, and
    signed smoothed minus filtered.
    """
    cfg = _bootstrap_cfg(bootstrap_n_replications=20)
    good, bad = [0.04, 0.06, 0.05, 0.05], [-0.04, -0.06, -0.05, -0.05]
    series = good + bad
    n = len(series)
    index = _months("2005-01-31", n)
    filt = pd.DataFrame({"label": [0, 1, 0, 1, 0, 1, 0, 1], "assigned": [True] * n}, index=index)
    smooth = pd.DataFrame({"label": [1, 1, 1, 1, 0, 0, 0, 0], "assigned": [True] * n}, index=index)
    factors = pd.DataFrame(
        {f: [0.0] + series for f in cfg.strategy_factors}, index=_months("2005-01-31", n + 1)
    )
    cfg = dataclasses.replace(cfg, sample_first_window_end="2005-01-31")

    gap = filtered_smoothed_gap(filt, smooth, factors, cfg)

    sqrt12 = np.sqrt(12.0)
    for row in gap.itertuples(index=False):
        filtered_sample = np.array(series[0::2] if row.state == 0 else series[1::2])
        smoothed_sample = np.array(bad if row.state == 0 else good)
        expected = (
            smoothed_sample.mean() / smoothed_sample.std(ddof=1) * sqrt12
            - filtered_sample.mean() / filtered_sample.std(ddof=1) * sqrt12
        )
        assert row.gap == pytest.approx(expected, abs=1e-12)
    assert (gap.loc[gap.state == 1, "gap"] > 0).all()
    assert (gap.loc[gap.state == 0, "gap"] < 0).all()


def test_excess_sharpe_is_conditional_minus_unconditional():
    """``excess_sharpe`` is the cell's Sharpe less that factor's pooled Sharpe, and nothing else.

    Reviewer answer to section 4 Q3: Mkt-RF pays 0.668 unconditionally over the
    out-of-sample window, so a conditional 0.712 is a level rather than a
    finding. The column is checked against a pooled table computed here from
    the same returns, not against the function that produced it, and the
    original nine columns are asserted unchanged and in place.
    """
    cfg = _bootstrap_cfg()
    rng = np.random.default_rng(cfg.run_seed)
    labels = list(rng.integers(0, 3, size=60))
    returns = {f: rng.normal(0.005, 0.03, size=60) for f in cfg.strategy_factors}
    label_frame, factors, fwe = _labels_and_factors(labels, returns, cfg.strategy_factors)
    cfg = dataclasses.replace(cfg, sample_first_window_end=fwe)

    stats = block_bootstrap_ci(label_frame, factors, cfg)
    pooled = unconditional_stats(label_frame, factors, cfg)
    with_excess = add_excess_sharpe(stats, pooled)

    assert list(with_excess.columns) == [
        "factor", "state", "n", "ann_mean", "ann_std", "sharpe", "excess_sharpe",
        "sharpe_p05", "sharpe_p95", "excludes_zero",
    ]
    pd.testing.assert_frame_equal(with_excess[list(stats.columns)], stats)

    sqrt12 = np.sqrt(12.0)
    for row in with_excess.itertuples(index=False):
        sample = np.array(returns[row.factor])                      # every month, no state, no assigned filter
        level = sample.mean() / sample.std(ddof=1) * sqrt12
        assert row.excess_sharpe == pytest.approx(row.sharpe - level, abs=1e-12)

    # a factor with no pooled row is an error, not a silent NaN column
    with pytest.raises(KeyError, match="unconditional_stats has no row"):
        add_excess_sharpe(stats, pooled.iloc[:1])


def test_refit_split_difference_zero_when_flags_identical():
    """When the two halves of the refit split hold identical returns, the difference is exactly 0.

    ``bootstrap_block_size`` is set far above the sample length, so every
    stationary-bootstrap draw is one circular block covering the whole series:
    a rotation, which preserves each (state, flag) half exactly. The two halves
    hold the same returns, so their Sharpes coincide in every replication and
    both percentiles of the difference are 0. That holds only because the two
    flags travel in the same bootstrap row; resampled independently the
    percentiles would straddle 0.

    The label series alternates 0, 1, 0, 1 ... so every row is its own run, and
    the run start is the row itself; the refit list then sets the flag row by
    row, giving each state six flagged and six unflagged months carrying the
    same six returns.
    """
    cfg = _bootstrap_cfg(bootstrap_block_size=10 ** 9, bootstrap_n_replications=50)
    values = [0.01, -0.02, 0.03, 0.005, -0.01, 0.02]

    labels, flags, returns = [], [], {f: [] for f in cfg.strategy_factors}
    for v in values:
        for flag in (False, True):
            for state in (0, 1):
                labels.append(state)
                flags.append(flag)
                for f in cfg.strategy_factors:
                    returns[f].append(v)
    label_frame, factors, fwe = _labels_and_factors(labels, returns, cfg.strategy_factors)
    cfg = dataclasses.replace(cfg, sample_first_window_end=fwe)
    refits = [d for d, flag in zip(label_frame.index, flags) if flag]

    assert list(run_started_on_refit(label_frame, refits)) == flags   # every row is its own run

    split, differences = bootstrap_refit_split(label_frame, factors, refits, cfg)

    assert list(split.columns) == [
        "factor", "state", "run_started_on_refit", "n", "ann_mean", "ann_std", "sharpe",
        "sharpe_p05", "sharpe_p95", "excludes_zero",
    ]
    assert list(differences.columns) == [
        "factor", "state", "sharpe_diff", "diff_p05", "diff_p95", "excludes_zero", "n_false", "n_true",
    ]
    assert len(split) == len(cfg.strategy_factors) * 2 * 2
    assert len(differences) == len(cfg.strategy_factors) * 2
    assert list(differences["n_false"]) == [6] * len(differences)
    assert list(differences["n_true"]) == [6] * len(differences)

    for row in differences.itertuples(index=False):
        assert row.sharpe_diff == pytest.approx(0.0, abs=1e-12)
        assert row.diff_p05 == pytest.approx(0.0, abs=1e-12)
        assert row.diff_p95 == pytest.approx(0.0, abs=1e-12)
        assert row.excludes_zero is False
