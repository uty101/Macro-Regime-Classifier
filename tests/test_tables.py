"""Step 2.4: the feature sanity table. Step 3.7: the per-refit HMM tables."""

import dataclasses

import numpy as np
import pandas as pd

from regime.config import load_config
from regime.tables import FEATURE_SANITY_COLUMNS, feature_sanity


def test_feature_sanity_shape(tmp_path) -> None:
    cfg = dataclasses.replace(load_config(), outputs_tables_dir=str(tmp_path))
    features = list(cfg.features_core) + list(cfg.features_robustness)
    index = pd.date_range("2001-01-31", "2003-12-31", freq="ME", name="date")
    rng = np.random.default_rng(cfg.run_seed)
    raw = pd.DataFrame(rng.normal(size=(len(index), len(features))), index=index, columns=features)
    # Hand-planted NaN: 2 in dgs10_level in 2001, 1 in log_vix in 2002, all 12 of breakeven_chg12 in 2003.
    raw.loc[["2001-03-31", "2001-11-30"], "dgs10_level"] = np.nan
    raw.loc["2002-07-31", "log_vix"] = np.nan
    raw.loc[index.year == 2003, "breakeven_chg12"] = np.nan

    table = feature_sanity(raw, cfg)
    assert list(table.columns) == list(FEATURE_SANITY_COLUMNS) == ["feature", "year", "min", "max", "n_nan"]
    assert len(table) == 10 * 3
    assert table["feature"].tolist() == [f for f in features for _ in range(3)]
    assert table["year"].tolist() == [2001, 2002, 2003] * 10

    by = table.set_index(["feature", "year"])
    assert by.loc[("dgs10_level", 2001), "n_nan"] == 2
    assert by.loc[("log_vix", 2002), "n_nan"] == 1
    assert by.loc[("breakeven_chg12", 2003), "n_nan"] == 12
    assert by["n_nan"].sum() == 2 + 1 + 12
    assert np.isnan(by.loc[("breakeven_chg12", 2003), "min"]) and np.isnan(by.loc[("breakeven_chg12", 2003), "max"])
    chunk = raw.loc[index.year == 2002, "oil_chg12"]
    assert by.loc[("oil_chg12", 2002), "min"] == chunk.min() and by.loc[("oil_chg12", 2002), "max"] == chunk.max()
    written = pd.read_csv(tmp_path / "feature_sanity.csv")
    assert written.shape == (30, 5)


def test_expected_duration_formula() -> None:
    from regime.tables import expected_durations

    # 1 / (1 - A_kk): 0.5 -> 2 months, 0.9 -> 10, 0.75 -> 4.
    transmat = np.array([[0.50, 0.30, 0.20], [0.05, 0.90, 0.05], [0.15, 0.10, 0.75]])

    np.testing.assert_allclose(expected_durations(transmat), [2.0, 10.0, 4.0], atol=1e-12)


def test_param_drift_shape(tmp_path) -> None:
    import dataclasses

    from regime.models.hmm import HMMParams
    from regime.tables import write_hmm_tables

    cfg = dataclasses.replace(
        load_config(), outputs_tables_dir=str(tmp_path)
    )
    features = list(cfg.features_core)
    K, d, refits = 3, len(features), [pd.Timestamp("2004-12-31"), pd.Timestamp("2005-12-31")]
    rng = np.random.default_rng(cfg.run_seed)
    params = [
        HMMParams(
            startprob=np.full(K, 1 / K),
            transmat=np.full((K, K), 1 / K),
            means=rng.normal(size=(K, d)),
            covars=np.stack([np.eye(d) * (k + 1) for k in range(K)]),
            K=K, refit_date=D, loglik=-1.0, converged=True,
        )
        for D in refits
    ]

    write_hmm_tables(params, features, cfg)
    drift = pd.read_csv(tmp_path / "param_drift.csv")

    assert list(drift.columns) == ["refit_date", "state", "feature", "mean", "variance"]
    assert len(drift) == len(refits) * K * d
    assert not drift.duplicated(subset=["refit_date", "state", "feature"]).any()
    assert set(drift["feature"]) == set(features)
    # variance is the diagonal of that state's anchored covariance, so state k
    # carries k + 1 on every feature here.
    for k in range(K):
        assert (drift.loc[drift["state"] == k, "variance"] == k + 1).all()
    assert len(pd.read_csv(tmp_path / "expected_duration.csv")) == len(refits) * K
    for D in refits:
        matrix = pd.read_csv(tmp_path / f"transition_matrix_{D:%Y-%m-%d}.csv", index_col="from_state")
        assert list(matrix.columns) == [f"to_{k}" for k in range(K)]
