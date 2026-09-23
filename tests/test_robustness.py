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

from regime.config import Config, load_config
from regime.models.hmm import hard_labels
from regime.run import ROBUSTNESS_SOURCES, robustness_cfg

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
