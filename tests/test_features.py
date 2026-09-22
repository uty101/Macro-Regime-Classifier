"""Section 2: raw feature definitions, row-based lags, the dollar splice and real-time standardisation."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from regime.config import load_config
from regime.data.asof import PANEL_COLUMNS
from regime.features import build_raw_features, model_input, splice_dollar_chg12, standardise


def _cfg(tmp_path):
    """The repo config with every section 2 output redirected to tmp_path."""
    cfg = load_config()
    return dataclasses.replace(
        cfg,
        outputs_features_raw=str(tmp_path / "features_raw.parquet"),
        outputs_features_z=str(tmp_path / "features_z.parquet"),
        outputs_dropped_rows=str(tmp_path / "dropped_rows.csv"),
    )


def _empty_panel(index: pd.DatetimeIndex) -> pd.DataFrame:
    panel = pd.DataFrame(np.nan, index=index, columns=list(PANEL_COLUMNS[:-3]), dtype="float64")
    for col in PANEL_COLUMNS[-3:]:
        panel[col] = pd.NaT
    panel.index.name = "date"
    return panel


# ---------------------------------------------------------------- step 2.1


def test_raw_feature_definitions_by_hand(tmp_path) -> None:
    """A 14-row panel of small integers; row 13 (the last) lags to row 1 with L = 12."""
    cfg = _cfg(tmp_path)
    assert cfg.features_change_lag == 12 and cfg.features_cpi_lag == 3
    # All 14 dates are on or after the splice date (2007-01-31), so dollar_chg12 is the DTWEXBGS change.
    index = pd.date_range("2007-01-31", periods=14, freq="ME", name="date")
    panel = _empty_panel(index)
    panel["dgs10"] = [7, 3] + [7] * 11 + [5]      # row 1 = 3, row 13 = 5
    panel["dgs2"] = [9] * 13 + [4]                 # row 13 = 4
    panel["dtwexbgs"] = [7, 100] + [7] * 11 + [120]  # row 1 = 100, row 13 = 120
    panel["dtwexm"] = [7, 100] + [7] * 11 + [300]  # would give ln 3 if the wrong series were used
    panel["wti"] = [7, 25] + [7] * 11 + [50]     # row 1 = 25, row 13 = 50
    panel["vix"] = [7] * 13 + [20]                 # row 13 = 20
    panel["t10yie"] = [7, 2] + [7] * 11 + [3]      # row 1 = 2, row 13 = 3
    panel["cpi_m"] = [7] * 13 + [104]
    panel["cpi_m3"] = [7] * 13 + [100]
    panel["indpro_m"] = [7] * 13 + [200]
    panel["indpro_m12"] = [7] * 13 + [100]
    panel["unrate_m"] = [7] * 13 + [6]
    panel["unrate_m12"] = [7] * 13 + [4]

    raw = build_raw_features(panel, cfg)
    assert list(raw.columns) == list(cfg.features_core) + list(cfg.features_robustness)
    assert raw.index.equals(index)
    last = raw.iloc[-1]
    expected = {
        "dgs10_level": 5.0,                        # dgs10
        "dgs10_chg12": 2.0,                        # 5 - 3
        "slope_2s10s": 1.0,                        # 5 - 4
        "cpi_3m_ann": 16.98585600000002,           # ((104/100)^4 - 1) * 100 = (1.16985856 - 1) * 100
        "indpro_chg12": 69.31471805599453,         # ln(200/100) * 100
        "dollar_chg12": 18.232155679395458,        # ln(120/100) * 100
        "oil_chg12": 69.31471805599453,            # ln(50/25) * 100
        "log_vix": 2.995732273553991,              # ln 20
        "unrate_chg12": 2.0,                       # 6 - 4
        "breakeven_chg12": 1.0,                    # 3 - 2
    }
    for col, value in expected.items():
        assert last[col] == pytest.approx(value, abs=1e-9), col
    assert (tmp_path / "features_raw.parquet").exists()


def test_lags_are_rows_not_days(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    index = pd.date_range("2000-01-31", periods=40, freq="ME", name="date")
    panel = _empty_panel(index)
    panel["dgs10"] = np.arange(len(index), dtype="float64")
    raw = build_raw_features(panel, cfg)
    chg = raw["dgs10_chg12"]
    assert chg.iloc[:12].isna().all()
    assert (chg.iloc[12:] == 12).all()


# ---------------------------------------------------------------- step 2.2


def test_dollar_splice_switches_exactly_at_splice_date(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    lag = cfg.features_change_lag
    splice = pd.Timestamp(cfg.features_dollar_splice_date)
    index = pd.date_range("2005-01-31", "2008-12-31", freq="ME", name="date")
    panel = _empty_panel(index)
    rng = np.random.default_rng(cfg.run_seed)
    panel["dtwexm"] = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, len(index))))
    panel["dtwexbgs"] = 50 * np.exp(np.cumsum(rng.normal(0, 0.02, len(index))))
    m = np.log(panel["dtwexm"] / panel["dtwexm"].shift(lag)) * 100
    b = np.log(panel["dtwexbgs"] / panel["dtwexbgs"].shift(lag)) * 100

    out = splice_dollar_chg12(panel, cfg)
    before = index[index.get_loc(splice) - 1]
    assert out.loc[before] == pytest.approx(m.loc[before], abs=1e-12)
    assert out.loc[before] != pytest.approx(b.loc[before], abs=1e-6)
    assert out.loc[splice] == pytest.approx(b.loc[splice], abs=1e-12)
    assert out.loc[splice] != pytest.approx(m.loc[splice], abs=1e-6)
    assert out.loc[index < splice].equals(m.loc[index < splice])
    assert out.loc[index >= splice].equals(b.loc[index >= splice])


def test_dollar_splice_has_no_level_jump(tmp_path) -> None:
    """Same path, different base: a change splice is seamless, a level splice would jump at the splice date."""
    cfg = _cfg(tmp_path)
    lag = cfg.features_change_lag
    index = pd.date_range("2005-01-31", "2008-12-31", freq="ME", name="date")
    panel = _empty_panel(index)
    rng = np.random.default_rng(cfg.run_seed)
    panel["dtwexbgs"] = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, len(index))))
    panel["dtwexm"] = 0.8 * panel["dtwexbgs"]
    expected = np.log(panel["dtwexbgs"] / panel["dtwexbgs"].shift(lag)) * 100

    out = splice_dollar_chg12(panel, cfg)
    assert out.isna().equals(expected.isna())
    assert out.iloc[:lag].isna().all() and out.iloc[lag:].notna().all()
    np.testing.assert_allclose(out.dropna().to_numpy(), expected.dropna().to_numpy(), rtol=0, atol=1e-12)


# ---------------------------------------------------------------- step 2.3


def _synthetic_raw(cfg) -> pd.DataFrame:
    """Ten random columns on the full decision-date index; breakeven_chg12 NaN before robustness_from."""
    rng = np.random.default_rng(cfg.run_seed)
    index = pd.date_range(pd.Timestamp(cfg.sample_start), pd.Timestamp(cfg.sample_end), freq="ME", name="date")
    cols = list(cfg.features_core) + list(cfg.features_robustness)
    raw = pd.DataFrame(rng.normal(size=(len(index), len(cols))), index=index, columns=cols)
    raw.loc[index < pd.Timestamp(cfg.sample_robustness_from), "breakeven_chg12"] = np.nan
    return raw


def _spike_rows(cfg) -> list:
    index = pd.date_range(pd.Timestamp(cfg.sample_features_from), pd.Timestamp(cfg.sample_end), freq="ME")
    w = index.get_loc(pd.Timestamp(cfg.sample_first_window_end))
    return [index[w], index[w + 1], index[w + 60], index[-2]]


# t inside the first window is not parametrised: for those dates z uses the moments of the whole
# window [features_from, first_window_end] by construction (convention 2), so a spike planted at t + 1
# inside the window changes them. From first_window_end onward nothing dated after t may move z at t.
@pytest.mark.parametrize("t", _spike_rows(load_config()), ids=lambda t: t.strftime("%Y-%m-%d"))
def test_spike_after_t_does_not_change_z_at_t(tmp_path, t) -> None:
    cfg = _cfg(tmp_path)
    raw = _synthetic_raw(cfg)
    z_clean = standardise(raw, cfg)

    window = raw.loc[pd.Timestamp(cfg.sample_features_from):pd.Timestamp(cfg.sample_first_window_end)]
    sigma_w = window.std(ddof=cfg.features_ddof)
    spiked = raw.copy()
    pos = spiked.index.get_loc(t)
    spiked.iloc[pos + 1] = spiked.iloc[pos + 1] + 10 * sigma_w
    z_spiked = standardise(spiked, cfg)

    upto = z_clean.index <= t
    np.testing.assert_allclose(
        z_spiked.loc[upto].to_numpy(), z_clean.loc[upto].to_numpy(), rtol=0, atol=1e-12, equal_nan=True
    )
    after = z_clean.index > t
    assert not np.allclose(z_spiked.loc[after].to_numpy(), z_clean.loc[after].to_numpy(), atol=1e-6, equal_nan=True)


def test_first_window_rule(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    raw = _synthetic_raw(cfg)
    z = standardise(raw, cfg)
    start = pd.Timestamp(cfg.sample_features_from)
    window_end = pd.Timestamp(cfg.sample_first_window_end)
    ddof = cfg.features_ddof
    assert z.index[0] == start and z.index[-1] == pd.Timestamp(cfg.sample_end)
    assert list(z.columns) == list(raw.columns)

    window = raw.loc[start:window_end]
    mean_w = window.mean()
    std_w = window.std(ddof=ddof)
    for t in [pd.Timestamp("1991-01-31"), pd.Timestamp("1997-06-30"), window_end]:
        expected = (raw.loc[t] - mean_w) / std_w
        np.testing.assert_allclose(z.loc[t].to_numpy(), expected.to_numpy(), rtol=0, atol=1e-12, equal_nan=True)

    t = pd.Timestamp("2005-01-31")
    upto = raw.loc[start:t]
    expected = (raw.loc[t] - upto.mean()) / upto.std(ddof=ddof)
    np.testing.assert_allclose(z.loc[t].to_numpy(), expected.to_numpy(), rtol=0, atol=1e-12)
    not_window = (raw.loc[t] - mean_w) / std_w
    assert not np.allclose(z.loc[t].to_numpy(), not_window.to_numpy(), atol=1e-6)

    # breakeven_chg12 is NaN before robustness_from: its window moments are over its own non-NaN rows.
    be = window["breakeven_chg12"].dropna()
    assert len(be) == 12
    t_be = be.index[-1]
    assert z.loc[t_be, "breakeven_chg12"] == pytest.approx((be.iloc[-1] - be.mean()) / be.std(ddof=ddof), abs=1e-12)
    assert z.loc[start:pd.Timestamp("2003-12-31"), "breakeven_chg12"].isna().all()


def test_model_input_drops_nan_rows_and_writes_them(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    raw = _synthetic_raw(cfg)
    raw.loc[pd.Timestamp("1990-06-30"), "log_vix"] = np.nan   # before features_from: never in the output
    raw.loc[pd.Timestamp("2010-03-31"), ["cpi_3m_ann", "oil_chg12"]] = np.nan
    z = standardise(raw, cfg)
    x = model_input(z, cfg)
    assert list(x.columns) == list(cfg.features_core)
    assert x.index[0] == pd.Timestamp(cfg.sample_features_from)
    assert pd.Timestamp("2010-03-31") not in x.index
    assert len(x) == len(z) - 1 and not x.isna().any().any()
    dropped = pd.read_csv(tmp_path / "dropped_rows.csv")
    assert list(dropped.columns) == ["date", "missing"]
    assert dropped["date"].tolist() == ["2010-03-31"]
    assert dropped["missing"].tolist() == ["cpi_3m_ann;oil_chg12"]
