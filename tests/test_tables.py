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


def _diagnostics(**d7) -> pd.DataFrame:
    """A two-row diagnostics table; d7 kwargs override the core_no_level row."""
    base = {
        "n_filtered_changes": 30, "n_changes_on_refit_dates": 5, "share_on_refit_dates": 0.17,
        "median_run_months": 6.0, "max_expected_duration": 80.0, "n_infinite_durations": 0,
        "n_degenerate_states": 0, "max_matched_distance": 0.8, "detects_2008": True,
        "detects_2020": True, "filtered_smoothed_agreement": 0.7,
    }
    core = dict(base, feature_set="core", n_changes_on_refit_dates=15)
    nolevel = dict(base, feature_set="core_no_level", **d7)
    return pd.DataFrame([core, nolevel])


def test_primary_feature_set_rule_needs_all_three_conditions() -> None:
    from regime.tables import primary_feature_set_decision

    # All three hold: d = 7 detects both episodes and makes fewer refit-date
    # changes than d = 8's 15.
    assert primary_feature_set_decision(_diagnostics(n_changes_on_refit_dates=5)) == "core_no_level"

    # Each condition alone is enough to send it back to core.
    assert primary_feature_set_decision(_diagnostics(detects_2008=False)) == "core"
    assert primary_feature_set_decision(_diagnostics(detects_2020=False)) == "core"
    assert primary_feature_set_decision(_diagnostics(n_changes_on_refit_dates=15)) == "core"  # not strictly below
    assert primary_feature_set_decision(_diagnostics(n_changes_on_refit_dates=16)) == "core"

    # No other column enters the rule: make every one of them worse for d = 7
    # while the three conditions hold, and it still wins.
    assert primary_feature_set_decision(
        _diagnostics(
            n_changes_on_refit_dates=5, median_run_months=1.0, n_degenerate_states=9,
            max_matched_distance=99.0, filtered_smoothed_agreement=0.0, n_infinite_durations=7,
        )
    ) == "core_no_level"


def test_label_runs_lengths_and_starts() -> None:
    from regime.tables import label_runs

    index = pd.date_range("2005-01-31", periods=7, freq="ME", name="date")
    labels = pd.Series([0, 0, 1, 1, 1, 0, 2], index=index)

    runs = label_runs(labels)

    assert list(runs["state"]) == [0, 1, 0, 2]
    assert list(runs["length"]) == [2, 3, 1, 1]
    assert list(runs["start"]) == [index[0], index[2], index[5], index[6]]
    assert runs["length"].sum() == len(labels)


# ---------------------------------------------------------------- step 7.2


def _label_frame(index, values, assigned=None) -> pd.DataFrame:
    frame = pd.DataFrame(
        {"label": np.asarray(values, dtype="int64"),
         "assigned": True if assigned is None else assigned},
        index=index,
    )
    frame.index.name = "date"
    return frame


def test_regime_labels_schema_and_coverage(tmp_path) -> None:
    from regime.tables import REGIME_LABEL_COLUMNS, write_regime_labels

    cfg = dataclasses.replace(load_config(), outputs_regime_labels=str(tmp_path / "regime_labels.csv"))
    full = pd.date_range(pd.Timestamp(cfg.sample_start), pd.Timestamp(cfg.sample_end), freq="ME", name="date")
    rng = np.random.default_rng(cfg.run_seed)

    # rules and smoothed run from features_from; filtered and the GMM only
    # from the first refit, which is what the real sources do.
    early = full[full >= pd.Timestamp(cfg.sample_features_from)]
    late = full[full >= pd.Timestamp(cfg.sample_first_window_end)]
    rules = _label_frame(early, rng.integers(0, 3, len(early)))
    smooth = _label_frame(early, rng.integers(0, 3, len(early)))
    assigned = rng.random(len(late)) > 0.2
    filt = _label_frame(late, rng.integers(0, 3, len(late)), assigned=assigned)
    gmm = _label_frame(late, rng.integers(0, 3, len(late)))

    table = write_regime_labels(rules, filt, smooth, gmm, cfg)

    assert list(table.columns) == list(REGIME_LABEL_COLUMNS)
    assert table.index.equals(full) and len(table) == len(full)

    # Empty exactly where the source has no row, and never elsewhere.
    assert table["hmm_filtered_label"].notna().sum() == len(late)
    assert table["gmm_filtered_label"].notna().sum() == len(late)
    assert table["rules_label"].notna().sum() == len(early)
    assert table["hmm_smoothed_label"].notna().sum() == len(early)
    assert table.loc[table.index < pd.Timestamp(cfg.sample_first_window_end), "hmm_filtered_label"].isna().all()

    # assigned is False on every date the filtered label is empty, and carries
    # the source's own flag everywhere else.
    unavailable = table["hmm_filtered_label"].isna()
    assert not table.loc[unavailable, "hmm_filtered_assigned"].any()
    assert table["hmm_filtered_assigned"].dtype == bool
    assert (table.loc[late, "hmm_filtered_assigned"].to_numpy() == assigned).all()

    written = pd.read_csv(tmp_path / "regime_labels.csv")
    assert list(written.columns) == ["date"] + list(REGIME_LABEL_COLUMNS)
    assert len(written) == len(full)
    assert written["date"].iloc[0] == str(full[0].date()) and written["date"].iloc[-1] == str(full[-1].date())


def test_check_outputs_lists_every_section_8_table(tmp_path) -> None:
    from regime.tables import CHECK_OUTPUTS_COLUMNS, SECTION_8_TABLES, check_outputs

    cfg = dataclasses.replace(
        load_config(),
        outputs_tables_dir=str(tmp_path / "tables"),
        outputs_regime_labels=str(tmp_path / "regimes" / "regime_labels.csv"),
    )
    tables = tmp_path / "tables"
    tables.mkdir()

    # Nothing written yet: every section 8 table is absent and not columns_ok.
    empty = check_outputs(cfg)
    assert list(empty.columns) == list(CHECK_OUTPUTS_COLUMNS)
    assert len(empty) == len(SECTION_8_TABLES) + 1  # + regime_labels.csv, no matrices yet
    assert not empty["present"].any() and not empty["columns_ok"].any()

    for name, columns in SECTION_8_TABLES.items():
        pd.DataFrame(columns=list(columns)).to_csv(tables / name, index=False)
    for refit in ("2004-12-31", "2005-12-31"):
        pd.DataFrame(
            [[1.0, 0.0, 0.0]], index=pd.Index([0], name="from_state"), columns=["to_0", "to_1", "to_2"]
        ).to_csv(tables / f"transition_matrix_{refit}.csv")
    labels = tmp_path / "regimes" / "regime_labels.csv"
    labels.parent.mkdir()
    from regime.tables import REGIME_LABEL_COLUMNS

    pd.DataFrame(columns=["date"] + list(REGIME_LABEL_COLUMNS)).to_csv(labels, index=False)

    full = check_outputs(cfg)
    assert len(full) == len(SECTION_8_TABLES) + 2 + 1
    assert full["present"].all() and full["columns_ok"].all()
    assert sum("transition_matrix_" in p for p in full["path"]) == 2
    for name in SECTION_8_TABLES:
        assert any(p.endswith(name) for p in full["path"]), name
    assert any(p.endswith("regime_labels.csv") for p in full["path"])

    # A table with the right name and the wrong columns is present, not ok.
    pd.DataFrame(columns=["K", "bic"]).to_csv(tables / "bic_by_k.csv", index=False)
    wrong = check_outputs(cfg).set_index("path")
    row = wrong.loc[(tables / "bic_by_k.csv").as_posix()]
    assert bool(row["present"]) and not bool(row["columns_ok"])
