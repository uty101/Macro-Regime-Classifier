"""Section 6: the robustness variants and the fragility of the headline statistics.

Every variant is a ``dataclasses.replace`` of the loaded config with its
output paths redirected under ``robustness/<name>/``. The first thing worth
proving about that is the thing a path bug would silently destroy: no variant
writes over a main output.

The tests that compare a variant's file with the main run's read committed
outputs. ``data/processed/`` is regenerated and never committed (convention
14), so nothing here reads it: the label source these tests need is the
committed ``outputs/regimes/filtered_probs.csv``, and the factor returns come
from the pinned French snapshot, which is committed in full.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from regime.config import Config, feature_set_columns, load_config, primary_columns
from regime.conditional import join_next_return
from regime.features import model_input
from regime.models.hmm import hard_labels, refit_dates
from regime.robustness import (
    _trim_count,
    leave_one_month_out,
    leave_one_month_out_pairwise,
    sharpe_difference,
    trimmed_count,
    trimmed_diff,
)
from regime.run import (
    ROBUSTNESS_SOURCES,
    headline_net_returns,
    nonprimary_feature_set,
    robustness_10feat_cfg,
    robustness_10feat_columns,
    robustness_cfg,
)
from regime.strategy import annualised_sharpe

OUTPUT_FIELDS = tuple(
    f.name for f in dataclasses.fields(Config)
    if f.name.startswith("outputs_") and f.type in ("str", str)
)


def _paths(cfg: Config) -> set:
    """Every ``outputs_*`` string field of a config, normalised to forward slashes."""
    return {getattr(cfg, name).replace("\\", "/") for name in OUTPUT_FIELDS}


def _read(path: str) -> pd.DataFrame:
    """A committed output CSV, or a failure naming it. Never a skip (PLAN.md's test convention)."""
    from pathlib import Path

    if not Path(path).exists():
        pytest.fail(f"{path} is absent; it is a committed output of section 6")
    return pd.read_csv(path)


def _headline_labels(cfg: Config) -> pd.DataFrame:
    """The published HMM filtered labels, from the committed probability file."""
    probs = pd.read_csv(cfg.outputs_filtered_probs, parse_dates=["date"]).set_index("date")
    probs.index = pd.DatetimeIndex(probs.index, name="date")
    return hard_labels(probs, cfg)


def _factors(cfg: Config) -> pd.DataFrame:
    from regime.data.french import load_french

    return load_french(cfg.french_pull_id, cfg)


def _synthetic_z(cfg: Config) -> pd.DataFrame:
    """A z matrix with every core and robustness column, over the whole decision-date index.

    Built in the test rather than read from ``data/processed/`` so a fresh
    clone runs it. The values are seeded noise; nothing here asserts anything
    about them, only about which columns and which dates survive.
    """
    rng = np.random.default_rng(cfg.run_seed)
    index = pd.date_range(cfg.sample_start, cfg.sample_end, freq="ME", name="date")
    columns = list(dict.fromkeys(list(cfg.features_core) + list(cfg.features_robustness)))
    return pd.DataFrame(rng.standard_normal((len(index), len(columns))), index=index, columns=columns)


# --------------------------------------------------------------- step 6.1


def test_k_variant_paths_do_not_collide_with_main_outputs() -> None:
    """Step 6.1: for each K, every redirected path differs from every main output path.

    Not merely "the tables directory differs": the whole ``outputs_*`` surface
    is compared, because a variant that redirected its tables but kept the
    main ``filtered_probs.csv`` would overwrite the published classifier and
    still look right in its own folder.
    """
    cfg = load_config()
    main = _paths(cfg)
    seen = {}
    for K in cfg.hmm_k_grid:
        vcfg = robustness_cfg(cfg, f"k{K}", strategy_label_sources=ROBUSTNESS_SOURCES)
        redirected = {
            name for name in OUTPUT_FIELDS
            if getattr(vcfg, name).replace("\\", "/") != getattr(cfg, name).replace("\\", "/")
        }
        assert redirected, f"K={K} redirected nothing"
        for name in redirected:
            assert getattr(vcfg, name).replace("\\", "/") not in main, (K, name)
            key = (name, getattr(vcfg, name).replace("\\", "/"))
            assert key not in seen, f"K={K} and K={seen[key]} share {name}"
            seen[key] = K
        assert vcfg.strategy_label_sources == ROBUSTNESS_SOURCES


def test_robustness_cfg_overrides_are_applied_last() -> None:
    """An override replaces a redirected path as well as a model parameter."""
    cfg = load_config()
    vcfg = robustness_cfg(cfg, "probe", hmm_covariance_type="diag", outputs_primary_k="x.txt")
    assert vcfg.hmm_covariance_type == "diag"
    assert vcfg.outputs_primary_k == "x.txt"
    assert vcfg.outputs_tables_dir.endswith("robustness/probe")
    assert cfg.hmm_covariance_type == "full"
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.hmm_covariance_type = "diag"  # type: ignore[misc]


def test_k_variant_outcomes_cover_every_k_in_the_grid() -> None:
    """Every K of ``hmm.k_grid`` is accounted for: completed, or named with its reason.

    K = 5 does not complete — its fitted covariance is rank-deficient at the
    2006-12-31 refit and ``forward_filter`` refuses it (``decisions/OPEN.md``
    item 1). That is recorded as an outcome rather than swallowed, and this
    test is what stops a K disappearing from step 6.1 silently.
    """
    cfg = load_config()
    outcomes = _read(f"{cfg.outputs_tables_dir}/robustness/k_variant_outcomes.csv")
    assert sorted(outcomes["K"]) == sorted(cfg.hmm_k_grid)
    for row in outcomes.itertuples(index=False):
        assert row.status in {"completed", "singular_covariance"}, row
        if row.status != "completed":
            assert str(row.detail).strip(), f"K={row.K} failed without a reason"
    completed = set(outcomes.loc[outcomes["status"] == "completed", "K"])
    summary = _read(f"{cfg.outputs_tables_dir}/robustness/robustness_summary.csv")
    assert {f"k{K}" for K in completed} <= set(summary["variant"])


SINGULAR_ERRORS = (
    np.linalg.LinAlgError("Matrix is not positive definite"),
    ValueError("component 4 of 'full' covars must be symmetric, positive-definite"),
)


def _k_variant_cfg(cfg: Config, tmp_path, K: int) -> Config:
    """``cfg`` with a one-K grid, a synthetic z on disk and step 6.1's outputs under ``tmp_path``.

    ``run_robustness_k`` reads ``outputs_features_z`` before the loop, and
    ``data/processed/`` is never committed, so the z it reads is written here
    (the variant never gets as far as using it: the fit is the thing being
    made to raise).
    """
    z_path = tmp_path / "features_z.parquet"
    _synthetic_z(cfg).to_parquet(z_path)
    return dataclasses.replace(
        cfg,
        hmm_k_grid=(K,),
        outputs_features_z=str(z_path),
        outputs_tables_dir=str(tmp_path / "tables"),
        outputs_regimes_dir=str(tmp_path / "regimes"),
        outputs_processed_dir=str(tmp_path / "processed"),
    )


@pytest.mark.parametrize("error", SINGULAR_ERRORS, ids=lambda e: type(e).__name__)
def test_singular_variant_is_recorded_not_raised(tmp_path, monkeypatch, error) -> None:
    """Step 6.1 records a rank-deficient K and carries on, whichever exception says so.

    ``forward_filter``'s Cholesky raises ``numpy.linalg.LinAlgError``; on
    another BLAS hmmlearn's ``_validate_covars`` finds a non-positive
    eigenvalue first and raises a plain ``ValueError``. The two are the same
    condition, and an uncaught one aborts ``python -m regime.run`` inside step
    6.1, so sections 6 and 7 never run.
    """
    from regime import run as run_module

    cfg = _k_variant_cfg(load_config(), tmp_path, 5)

    def _raise(*args, **kwargs):
        raise error

    monkeypatch.setattr(run_module, "variant_classifier", _raise)
    run_module.run_robustness_k(cfg)                      # returns: the run continues

    outcomes = _read(f"{cfg.outputs_tables_dir}/robustness/k_variant_outcomes.csv")
    assert list(outcomes.columns) == list(run_module.K_OUTCOME_COLUMNS)
    assert outcomes["K"].tolist() == [5]
    row = outcomes.iloc[0]
    assert row["status"] == "singular_covariance"
    assert row["detail"] == str(error)
    print(outcomes.to_string(index=False))


def test_a_variant_failure_that_is_not_singular_is_not_swallowed(tmp_path, monkeypatch) -> None:
    """Every other exception stays uncaught, including every other ``ValueError``."""
    from regime import run as run_module

    cfg = _k_variant_cfg(load_config(), tmp_path, 5)

    def _raise(*args, **kwargs):
        raise ValueError("model_input: column dgs10_level is missing")

    monkeypatch.setattr(run_module, "variant_classifier", _raise)
    with pytest.raises(ValueError, match="column dgs10_level is missing"):
        run_module.run_robustness_k(cfg)


# --------------------------------------------------------------- step 6.2


def test_diag_fit_covars_are_diagonal_full_shape() -> None:
    """Step 6.2: a ``diag`` fit still returns (K, d, d) covariances, with zero off-diagonals.

    This is the whole reason ``forward_filter`` and the anchoring need no
    change for the diagonal variant: hmmlearn's ``covars_`` property returns
    full-shaped matrices for every covariance type.
    """
    cfg = load_config()
    vcfg = robustness_cfg(cfg, "diag", hmm_covariance_type="diag")
    rng = np.random.default_rng(cfg.run_seed)
    K, d, T = 2, 4, 200
    means = np.array([[-2.0] * d, [2.0] * d])
    draw = rng.standard_normal((T, d)) + means[rng.integers(0, K, T)]

    from regime.models.hmm import fit_hmm

    params, _restarts = fit_hmm(draw, K, vcfg, pd.Timestamp("2004-12-31"))
    assert params.covars.shape == (K, d, d)
    for k in range(K):
        off = params.covars[k] - np.diag(np.diag(params.covars[k]))
        assert np.allclose(off, 0.0), params.covars[k]
        assert (np.diag(params.covars[k]) > 0).all()


def test_diag_covars_round_trip_through_the_hmmlearn_setter() -> None:
    """Step 6.2: a ``diag`` parameter set survives ``model_from_params`` and can be filtered.

    hmmlearn's ``covars_`` getter returns (K, d, d) for every covariance type
    but its setter rejects that shape for ``"diag"``. ``HMMParams`` holds one
    shape throughout, so the conversion happens in exactly one place, and this
    is what proves the two halves agree.
    """
    from regime.models.hmm import HMMParams, covars_for_setter, model_from_params
    from regime.models.hmm_numpy import forward_filter

    cfg = load_config()
    vcfg = robustness_cfg(cfg, "diag", hmm_covariance_type="diag")
    K, d = 3, 4
    covars = np.array([np.diag(np.full(d, 0.5 + k)) for k in range(K)])
    params = HMMParams(
        startprob=np.full(K, 1.0 / K),
        transmat=np.full((K, K), 1.0 / K),
        means=np.arange(K * d, dtype="float64").reshape(K, d),
        covars=covars,
        K=K,
        refit_date=pd.Timestamp("2004-12-31"),
        loglik=0.0,
        converged=True,
    )
    assert covars_for_setter(covars, "diag").shape == (K, d)
    assert covars_for_setter(covars, "full").shape == (K, d, d)
    model = model_from_params(params, vcfg)
    # hmmlearn's covars_ getter needs n_features for the non-full types, which
    # only fitting or a forward pass sets; _covars_ is what the setter stored.
    assert np.allclose(np.asarray(model._covars_), np.diagonal(covars, axis1=1, axis2=2))

    rng = np.random.default_rng(cfg.run_seed)
    y = rng.standard_normal((20, d))
    posterior = model.predict_proba(y)
    assert posterior.shape == (20, K)
    assert np.allclose(posterior.sum(axis=1), 1.0)
    alpha, _loglik = forward_filter(y, params.startprob, params.transmat, params.means, params.covars)
    assert alpha.shape == (20, K)
    assert np.allclose(alpha.sum(axis=1), 1.0)

    with pytest.raises(NotImplementedError, match="tied"):
        covars_for_setter(covars, "tied")
    with pytest.raises(ValueError, match="off-diagonal"):
        dense = covars.copy()
        dense[0, 0, 1] = 0.3
        covars_for_setter(dense, "diag")


# --------------------------------------------------------------- step 6.3


def test_10feat_model_input_has_ten_columns_from_2004_01_31(tmp_path) -> None:
    """Step 6.3: the variant's model input is the ten ``core + robustness`` columns, from 2004-01-31."""
    cfg = load_config()
    vcfg = dataclasses.replace(
        robustness_10feat_cfg(cfg), outputs_dropped_rows=str(tmp_path / "dropped.csv")
    )
    columns = robustness_10feat_columns(cfg)
    assert len(columns) == 10
    assert columns == tuple(cfg.features_core) + tuple(cfg.features_robustness)
    assert len(set(columns)) == 10

    x = model_input(_synthetic_z(cfg), vcfg, columns=columns)
    assert list(x.columns) == list(columns)
    assert x.index[0] == pd.Timestamp(cfg.sample_robustness_from) == pd.Timestamp("2004-01-31")
    assert vcfg.hmm_covariance_type == cfg.hmm_robustness_covariance_type == "diag"


def test_10feat_refit_dates_start_2009_12_31() -> None:
    """Step 6.3: the variant refits from 2009-12-31, every December, and never before."""
    cfg = load_config()
    dates = refit_dates(robustness_10feat_cfg(cfg))
    assert dates[0] == pd.Timestamp("2009-12-31") == pd.Timestamp(cfg.hmm_robustness_first_refit)
    assert all(d.month == 12 and d.day == 31 for d in dates)
    assert all(b.year - a.year == 1 for a, b in zip(dates, dates[1:]))
    assert dates[0] > refit_dates(cfg)[0]


# --------------------------------------------------------------- step 6.4


def test_minobs_24_reproduces_main_grid() -> None:
    """Step 6.4: the N = 24 file is ``timing_results.csv``. 24 is the configured value."""
    cfg = load_config()
    assert cfg.strategy_min_regime_obs == 24
    main = _read(cfg.outputs_timing_results)
    variant = _read(f"{cfg.outputs_tables_dir}/timing_results_minobs24.csv")
    pd.testing.assert_frame_equal(main, variant)


def test_minobs_fallback_share_falls_as_min_regime_obs_falls() -> None:
    """Step 6.4: a lower threshold cannot put more months into fallback.

    The thin-regime branch fires on ``n_k(t) < min_regime_obs``, so lowering
    the threshold can only remove months from it; the other two branches do
    not depend on it at all. A table that said otherwise would mean the
    threshold was not the only thing that moved.
    """
    cfg = load_config()
    table = _read(f"{cfg.outputs_tables_dir}/robustness/minobs_fallback.csv")
    assert sorted(table["min_regime_obs"].unique()) == sorted(cfg.strategy_min_regime_obs_grid)
    for (source, eta), group in table.groupby(["source", "eta"]):
        ordered = group.sort_values("min_regime_obs")
        shares = ordered["fallback_share"].to_numpy(dtype="float64")
        assert (np.diff(shares) >= -1e-12).all(), (source, eta, ordered.to_string(index=False))


# --------------------------------------------------------------- step 6.5


def test_blocksize_6_reproduces_headline_row() -> None:
    """Step 6.5: b = 6 is the configured block size, so that row is the headline row."""
    cfg = load_config()
    assert cfg.bootstrap_block_size == 6
    sweep = _read(f"{cfg.outputs_tables_dir}/timing_results_blocksize.csv")
    row = sweep.loc[sweep["block_size"] == cfg.bootstrap_block_size]
    assert len(row) == 1
    row = row.iloc[0]

    grid = _read(cfg.outputs_timing_results)
    headline = grid.loc[
        (grid["eta"] == cfg.strategy_headline_eta)
        & (grid["lag"] == cfg.strategy_headline_lag)
        & (grid["cost_bp"] == cfg.strategy_headline_cost_bp)
        & (grid["source"] == cfg.strategy_headline_source)
    ]
    assert len(headline) == 1
    headline = headline.iloc[0]
    for column in ("sharpe_static", "sharpe_timed", "diff", "diff_p05", "diff_p95",
                   "p_one_sided", "mean_turnover"):
        assert float(row[column]) == pytest.approx(float(headline[column]), abs=1e-12), column
    assert int(row["n_months"]) == int(headline["n_months"])


# --------------------------------------------------------------- step 6.6


def test_nonprimary_model_input_is_the_other_feature_set(tmp_path) -> None:
    """Step 6.6: the rerun uses whichever set ``features.primary`` does not name, and never the main one."""
    cfg = load_config()
    other = nonprimary_feature_set(cfg)
    assert other != cfg.features_primary
    columns = feature_set_columns(cfg, other)
    assert tuple(columns) != tuple(primary_columns(cfg))

    vcfg = dataclasses.replace(
        robustness_cfg(cfg, "probe"), outputs_dropped_rows=str(tmp_path / "dropped.csv")
    )
    z = _synthetic_z(cfg)
    x = model_input(z, vcfg, columns=tuple(columns))
    assert list(x.columns) == list(columns)
    assert list(x.columns) != list(primary_columns(cfg))
    # the non-primary set is the core eight minus dgs10_level, not a different eight
    assert set(columns) < set(primary_columns(cfg)) or set(primary_columns(cfg)) < set(columns)


# --------------------------------------------------------------- step 6.7


def _pair_series(n: int, seed: int) -> tuple[pd.Series, pd.Series]:
    """Two net-return series on one index of ``n`` month-ends."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2005-01-31", periods=n, freq="ME", name="date")
    return (
        pd.Series(rng.normal(0.004, 0.02, n), index=index, name="timed"),
        pd.Series(rng.normal(0.003, 0.02, n), index=index, name="static"),
    )


def test_loo_returns_one_row_per_month() -> None:
    """Step 6.7: one row per earning month, in order, each dropping exactly that month."""
    cfg = load_config()
    timed, static = _pair_series(60, cfg.run_seed)
    loo = leave_one_month_out(timed, static, cfg)

    assert list(loo.columns) == ["dropped_month", "diff", "sign_flipped"]
    assert len(loo) == len(timed) == 60
    assert list(loo["dropped_month"]) == list(timed.index)
    assert loo["diff"].notna().all()

    # each row is the statistic over the other 59 months, not over all 60
    full = sharpe_difference(timed.to_numpy(), static.to_numpy(), cfg)
    for position in (0, 17, 59):
        keep = timed.index != timed.index[position]
        expected = sharpe_difference(
            timed[keep].to_numpy(), static[keep].to_numpy(), cfg
        )
        assert float(loo["diff"].iloc[position]) == pytest.approx(expected, abs=1e-12)
    assert not np.isclose(loo["diff"].to_numpy(), full, atol=1e-15).all()


def test_loo_full_value_matches_the_grid() -> None:
    """Step 6.7: the un-dropped statistic equals ``timing_results.csv``'s ``diff`` to 1e-12.

    The point of step 6.7 is to perturb the *grid's* number. If the series it
    perturbs were built differently from the ones the grid used, every LOO row
    would be a fragility of something else, and nothing else in the section
    would notice.
    """
    cfg = load_config()
    labels = _headline_labels(cfg)
    factors = _factors(cfg)
    timed, static = headline_net_returns(cfg, {cfg.strategy_headline_source: labels}, factors)
    full = sharpe_difference(timed.to_numpy(dtype="float64"), static.to_numpy(dtype="float64"), cfg)

    grid = _read(cfg.outputs_timing_results)
    headline = grid.loc[
        (grid["eta"] == cfg.strategy_headline_eta)
        & (grid["lag"] == cfg.strategy_headline_lag)
        & (grid["cost_bp"] == cfg.strategy_headline_cost_bp)
        & (grid["source"] == cfg.strategy_headline_source)
    ].iloc[0]
    assert full == pytest.approx(float(headline["diff"]), abs=1e-12)
    assert len(timed) == int(headline["n_months"])

    summary = _read(f"{cfg.outputs_tables_dir}/robustness/robustness_summary.csv")
    assert "headline_diff" not in set(summary["variant"]), "6.7 is a diagnostic, not a variant"
    fragility = _read(f"{cfg.outputs_tables_dir}/robustness/fragility_summary.csv")
    row = fragility.loc[fragility["statistic"] == "headline_diff"].iloc[0]
    assert float(row["full_value"]) == pytest.approx(full, abs=1e-12)
    assert int(row["n_months"]) == len(timed)


def test_trimmed_diff_drops_the_right_count_from_both_series() -> None:
    """Step 6.7: the trim removes ``floor(0.05 n)`` months, the same ones from both series."""
    cfg = load_config()
    n = 257
    timed, static = _pair_series(n, cfg.run_seed)
    count = trimmed_count(timed, static)
    assert count == _trim_count(n, 0.05) == 12
    assert count / n <= 0.05

    influence = (timed - static).abs()
    dropped = influence.sort_values(ascending=False).index[:count]
    keep = ~timed.index.isin(dropped)
    assert int(keep.sum()) == n - count

    expected = (
        annualised_sharpe(timed[keep].to_numpy(), cfg.features_ddof)
        - annualised_sharpe(static[keep].to_numpy(), cfg.features_ddof)
    )
    assert trimmed_diff(timed, static, cfg=cfg) == pytest.approx(expected, abs=1e-12)

    # both series lose the same months: trimming only one would leave the two
    # Sharpes computed over different samples, which is the error this guards
    assert len(timed[keep]) == len(static[keep])
    assert trimmed_diff(timed, static, trim=0.0, cfg=cfg) == pytest.approx(
        sharpe_difference(timed.to_numpy(), static.to_numpy(), cfg), abs=1e-12
    )


def test_pairwise_loo_covers_the_months_of_both_states() -> None:
    """Step 6.7: the pairwise table has one row per assigned month in either state, and no other."""
    cfg = load_config()
    labels = _headline_labels(cfg)
    factors = _factors(cfg)
    fragility = _read(f"{cfg.outputs_tables_dir}/robustness/fragility_summary.csv")
    pairs = fragility.loc[fragility["statistic"] != "headline_diff"]
    assert len(pairs) >= 1, "section 4 found a pairwise difference excluding zero; 6.7 must cover it"

    differences = _read(
        f"{cfg.outputs_tables_dir}/conditional_differences_{cfg.strategy_headline_source}.csv"
    )
    hit = differences.loc[differences["excludes_zero"].astype(bool)].iloc[0]
    loo = leave_one_month_out_pairwise(
        labels, factors, hit.factor, int(hit.state_a), int(hit.state_b), cfg
    )
    joined = join_next_return(labels, factors, cfg)
    expected = joined.loc[
        joined["assigned"] & joined["label"].isin([int(hit.state_a), int(hit.state_b)])
    ]
    assert len(loo) == len(expected) == int(hit.n_a) + int(hit.n_b)
    assert list(loo["dropped_month"]) == list(expected.index)
