"""Step 2.4: the feature sanity table."""

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
