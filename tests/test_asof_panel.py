"""Step 1.7: the panel schema, no use of planted future market values, and lags from the same vintage."""

import dataclasses

import numpy as np
import pandas as pd

from regime.config import load_config
from regime.data.alfred import asof_from_releases
from regime.data.asof import MARKET_COLUMNS, PANEL_COLUMNS, REVISED, assemble_panel, build_asof_panel
from regime.data.fred import month_end_from_daily

EXPECTED = [
    "dgs10", "dgs2", "dtwexbgs", "dtwexm", "wti", "vix", "t10yie",
    "cpi_m", "cpi_m3", "indpro_m", "indpro_m12", "unrate_m", "unrate_m12",
    "cpi_obs_month", "indpro_obs_month", "unrate_obs_month",
]


def test_schema_matches_section_10() -> None:
    cfg = load_config()
    panel = build_asof_panel(cfg)
    assert list(panel.columns) == EXPECTED == list(PANEL_COLUMNS)
    assert panel.index.name == "date"
    assert panel.index.is_month_end.all()
    assert panel.index.is_monotonic_increasing and panel.index.is_unique
    assert panel.index[0] == pd.Timestamp(cfg.sample_start)
    assert panel.index[-1] == pd.Timestamp(cfg.sample_end)
    for col in EXPECTED[-3:]:
        assert str(panel[col].dtype) == "datetime64[ns]", col
    for col in EXPECTED[:-3]:
        assert panel[col].dtype == "float64", col


def _synthetic_inputs(cfg, plant: pd.Timestamp | None = None):
    rng = np.random.default_rng(cfg.run_seed)
    days = pd.date_range("2000-01-01", "2002-12-31", freq="D")
    market = {}
    for sid in MARKET_COLUMNS:
        daily = pd.Series(rng.normal(size=len(days)), index=days, name=sid)
        if plant is not None and sid == "DGS10":
            daily.loc[plant] = 1e6
        market[sid] = month_end_from_daily(daily, pd.Timestamp(cfg.sample_start), cfg.fred_lookback_days)
    releases = []
    for m in pd.date_range("1999-01-01", "2002-12-01", freq="MS"):
        first = m + pd.offsets.MonthEnd(0) + pd.Timedelta(days=15)
        for rs in (first, first + pd.DateOffset(months=6)):
            releases.append({"date": m, "realtime_start": rs, "value": float(m.year * 100 + m.month) + rs.toordinal() * 1e-7})
    rel = pd.DataFrame(releases)
    dd = pd.date_range(pd.Timestamp(cfg.sample_start), pd.Timestamp(cfg.sample_end), freq="ME")
    asof = {sid: asof_from_releases(rel, dd) for sid in REVISED}
    return market, asof


def test_planted_future_market_value_is_ignored() -> None:
    cfg = dataclasses.replace(load_config(), sample_start="2000-01-31", sample_end="2002-11-30")
    t = pd.Timestamp("2001-06-30")
    clean = assemble_panel(*_synthetic_inputs(cfg), cfg)
    planted = assemble_panel(*_synthetic_inputs(cfg, plant=t + pd.Timedelta(days=1)), cfg)
    upto = clean.index <= t
    num = [c for c in EXPECTED if not c.endswith("_obs_month")]
    diff = (planted.loc[upto, num] - clean.loc[upto, num]).abs().fillna(0.0)
    assert (diff <= 1e-12).all().all(), diff.max()
    assert planted.loc[upto, EXPECTED[-3:]].equals(clean.loc[upto, EXPECTED[-3:]])
    # the plant sits on t + 1 day inside the synthetic DGS10 daily series; rows <= t must not see it
    assert 1e6 not in clean["dgs10"].to_numpy()


def test_lags_come_from_same_vintage() -> None:
    cfg = dataclasses.replace(load_config(), sample_start="2000-01-31", sample_end="2002-11-30")
    # value encodes (obs_month, realtime_start): int part = YYYYMM, fractional part = ordinal(realtime_start) * 1e-7
    market, asof = _synthetic_inputs(cfg)
    panel = assemble_panel(market, asof, cfg)
    rel_dates = sorted({pd.Timestamp(r) for r in asof["CPIAUCSL"].dropna()["value"].map(lambda v: round((v % 1) / 1e-7)).map(pd.Timestamp.fromordinal)})
    for t in panel.index:
        m, m3, obs = panel.loc[t, "cpi_m"], panel.loc[t, "cpi_m3"], panel.loc[t, "cpi_obs_month"]
        assert int(m) == obs.year * 100 + obs.month
        m3_month = obs - pd.offsets.MonthEnd(cfg.features_cpi_lag)
        assert int(m3) == m3_month.year * 100 + m3_month.month
        for v, month_start in ((m, obs - pd.offsets.MonthBegin(1)), (m3, m3_month - pd.offsets.MonthBegin(1))):
            rs = pd.Timestamp.fromordinal(round((v % 1) / 1e-7))
            assert rs <= t, (t, v, rs)
            eligible = pd.date_range(month_start + pd.offsets.MonthEnd(0) + pd.Timedelta(days=15), periods=2, freq=pd.DateOffset(months=6))
            assert rs == max(d for d in eligible if d <= t), (t, v, rs)
    assert len(rel_dates) > 0
