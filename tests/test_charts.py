"""Step 2.4: the features review chart. Step 7.1: the three section 7 charts."""

import numpy as np
import pandas as pd

from regime.charts import (
    conditional_sharpe_heatmap,
    features_review_chart,
    regimes_timeline,
    state_legend_labels,
    timed_vs_static,
)
from regime.config import load_config


def test_features_review_chart_written(tmp_path) -> None:
    cfg = load_config()
    features = list(cfg.features_core) + list(cfg.features_robustness)
    index = pd.date_range(pd.Timestamp(cfg.sample_start), pd.Timestamp(cfg.sample_end), freq="ME", name="date")
    rng = np.random.default_rng(cfg.run_seed)
    raw = pd.DataFrame(rng.normal(size=(len(index), len(features))), index=index, columns=features)
    path = tmp_path / "charts" / "features_review.png"
    features_review_chart(raw, str(path), cfg)
    assert path.exists()
    assert path.stat().st_size > 10_000


def _synthetic_inputs(cfg):
    """A panel, two label frames, a param_drift table and three backtests, all synthetic."""
    index = pd.date_range(
        pd.Timestamp(cfg.sample_features_from), pd.Timestamp(cfg.sample_end), freq="ME", name="date"
    )
    rng = np.random.default_rng(cfg.run_seed)
    panel = pd.DataFrame(
        {"dgs10_level": rng.normal(size=len(index)), "cpi_3m_ann": rng.normal(size=len(index))},
        index=index,
    )
    # Three states in long runs, so the shading has spans rather than stripes.
    states = np.repeat(np.arange(len(index) // 24 + 1) % 3, 24)[: len(index)]
    filt = pd.DataFrame({"label": states, "assigned": True}, index=index)
    smooth = pd.DataFrame({"label": np.roll(states, 3), "assigned": True}, index=index)

    features = list(cfg.features_core)
    drift = pd.DataFrame(
        [
            {"refit_date": D, "state": k, "feature": f, "mean": float(k) + j / 100, "variance": 1.0}
            for D in ("2004-12-31", "2025-12-31")
            for k in range(3)
            for j, f in enumerate(features)
        ]
    )
    backtests = {
        name: pd.DataFrame({"net_ret": rng.normal(0.004, 0.02, size=len(index))}, index=index)
        for name in ("static", "timed_filtered", "timed_smoothed")
    }
    return panel, filt, smooth, drift, backtests


def _synthetic_stats(cfg, excludes: set) -> pd.DataFrame:
    """A factor x state conditional-stats frame; ``excludes`` names the bordered cells."""
    rng = np.random.default_rng(cfg.run_seed)
    rows = []
    for factor in cfg.strategy_factors:
        for state in range(3):
            rows.append(
                {
                    "factor": factor, "state": state, "n": 40 + state,
                    "ann_mean": 0.03, "ann_std": 0.1, "sharpe": float(rng.normal()),
                    "excess_sharpe": 0.0, "sharpe_p05": -1.0, "sharpe_p95": 1.0,
                    "excludes_zero": (factor, state) in excludes,
                }
            )
    return pd.DataFrame(rows)


def test_three_charts_written_from_synthetic_inputs(tmp_path) -> None:
    cfg = load_config()
    panel, filt, smooth, drift, backtests = _synthetic_inputs(cfg)

    timeline = tmp_path / "charts" / "regimes_timeline.png"
    regimes_timeline(panel, filt, smooth, drift, str(timeline), cfg)

    heatmap = tmp_path / "charts" / "conditional_sharpe_filtered.png"
    conditional_sharpe_heatmap(_synthetic_stats(cfg, {("SMB", 1)}), str(heatmap), cfg)

    curves = tmp_path / "charts" / "timed_vs_static.png"
    timed_vs_static(backtests, str(curves), cfg)

    for path in (timeline, heatmap, curves):
        assert path.exists(), path
        assert path.stat().st_size > 10_000, path


def test_heatmap_border_only_where_excludes_zero(tmp_path) -> None:
    cfg = load_config()
    # Four cells excluding zero out of 6 factors x 3 states = 18.
    excludes = {("Mkt-RF", 0), ("SMB", 2), ("UMD", 1), ("CMA", 0)}
    stats = _synthetic_stats(cfg, excludes)
    assert int(stats["excludes_zero"].sum()) == len(excludes) == 4

    ax = conditional_sharpe_heatmap(stats, str(tmp_path / "heatmap.png"), cfg)

    borders = [p for p in ax.patches if not p.get_fill()]
    assert len(borders) == len(ax.patches) == int(stats["excludes_zero"].sum())
    factors = list(cfg.strategy_factors)
    placed = {(factors[int(round(p.get_y() + 0.5))], int(round(p.get_x() + 0.5))) for p in borders}
    assert placed == excludes

    # No excludes_zero row, no patches at all.
    ax_none = conditional_sharpe_heatmap(_synthetic_stats(cfg, set()), str(tmp_path / "none.png"), cfg)
    assert list(ax_none.patches) == []


def test_state_legend_labels_read_the_last_refit(tmp_path) -> None:
    cfg = load_config()
    _panel, _f, _s, drift, _b = _synthetic_inputs(cfg)
    # The 2025-12-31 rows carry mean = k + j/100 for feature j, as does 2004-12-31
    # here; make the last refit distinguishable so the function cannot pass by
    # reading the first one.
    last = pd.to_datetime(drift["refit_date"]) == pd.Timestamp("2025-12-31")
    drift.loc[last, "mean"] = drift.loc[last, "mean"] + 10.0

    labels = state_legend_labels(drift, cfg)

    assert set(labels) == {0, 1, 2}
    anchor_j = list(cfg.features_core).index(cfg.hmm_anchor_feature)
    tie_j = list(cfg.features_core).index(cfg.hmm_anchor_tiebreak_feature)
    for k in (0, 1, 2):
        expected = (
            f"state {k}: mean {cfg.hmm_anchor_feature} = {k + anchor_j / 100 + 10.0:.2f}, "
            f"mean {cfg.hmm_anchor_tiebreak_feature} = {k + tie_j / 100 + 10.0:.2f}"
        )
        assert labels[k] == expected
