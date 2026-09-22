"""Step 1.4: the pinned vintage files start before 1995 and the as-of rule never uses a release after t."""

import numpy as np
import pandas as pd

from regime.config import load_config
from regime.data.alfred import asof_from_releases, asof_path, build_asof, load_releases


def test_pinned_vintages_start_before_1995() -> None:
    cfg = load_config()
    for series_id in cfg.fred_vintage_series:
        rel = load_releases(series_id, cfg.fred_vintage_pull_id, cfg)
        earliest = pd.to_datetime(rel["realtime_start"]).min()
        assert earliest < pd.Timestamp("1995-01-01"), (series_id, earliest)


def _synthetic_releases() -> pd.DataFrame:
    # Observation months 2000-01 .. 2000-12; each has three releases: first release
    # ~15 days after the month, then +1 month and +6 months. value = ordinal day of
    # realtime_start, so the value identifies the release it came from.
    rows = []
    for m in pd.date_range("2000-01-01", "2000-12-01", freq="MS"):
        first = m + pd.offsets.MonthEnd(0) + pd.Timedelta(days=15)
        for rs in (first, first + pd.DateOffset(months=1), first + pd.DateOffset(months=6)):
            rows.append({"date": m, "realtime_start": rs, "value": float(rs.toordinal())})
    return pd.DataFrame(rows)


def test_value_used_has_realtime_start_le_t() -> None:
    releases = _synthetic_releases()
    decision_dates = pd.date_range("2000-01-31", "2001-06-30", freq="ME")
    out = asof_from_releases(releases, decision_dates)

    assert list(out.columns) == ["decision_date", "obs_month", "value"]
    for t in decision_dates:
        sub = out[out["decision_date"] == t]
        assert (sub["obs_month"] <= t).all()
        # itertuples keeps the float64 dtype of value (iterrows would upcast a NaN to NaT)
        for row in sub.itertuples(index=False):
            m_start = row.obs_month - pd.offsets.MonthBegin(1)
            eligible = releases[(releases["date"] == m_start) & (releases["realtime_start"] <= t)]
            if len(eligible) == 0:
                assert np.isnan(row.value), (t, row.obs_month, row.value)
            else:
                assert row.value <= t.toordinal()
                assert row.value == float(eligible["realtime_start"].max().toordinal())

    # 2000-01 is first released on 2000-02-15, so it is NaN at 2000-01-31 and present at 2000-02-29.
    jan = out[(out["obs_month"] == pd.Timestamp("2000-01-31"))].set_index("decision_date")["value"]
    assert np.isnan(jan.loc[pd.Timestamp("2000-01-31")])
    assert jan.loc[pd.Timestamp("2000-02-29")] == float(pd.Timestamp("2000-02-15").toordinal())
    # its +6 month revision (2000-08-15) shows from 2000-08-31 onward, never before
    assert jan.loc[pd.Timestamp("2000-07-31")] == float(pd.Timestamp("2000-03-15").toordinal())
    assert jan.loc[pd.Timestamp("2000-08-31")] == float(pd.Timestamp("2000-08-15").toordinal())


def test_no_release_before_t_gives_all_nan() -> None:
    releases = _synthetic_releases()
    out = asof_from_releases(releases, pd.DatetimeIndex([pd.Timestamp("1999-12-31"), pd.Timestamp("2000-01-31")]))
    assert out["value"].isna().all()


def test_cpi_obs_month_is_t_minus_1() -> None:
    cfg = load_config()
    path = asof_path("CPIAUCSL", cfg.fred_vintage_pull_id, cfg)
    table = pd.read_parquet(path) if path.exists() else build_asof("CPIAUCSL", cfg.fred_vintage_pull_id, cfg)
    latest = table.dropna(subset=["value"]).groupby("decision_date")["obs_month"].max()
    all_t = pd.DatetimeIndex(sorted(table["decision_date"].unique()))
    latest = latest.reindex(all_t)
    expected = all_t - pd.offsets.MonthEnd(1)
    ok = latest.to_numpy() == expected.to_numpy()
    share = ok.mean()
    failing = pd.DataFrame({"decision_date": all_t[~ok], "latest_obs_month": latest[~ok].to_numpy()})
    assert share >= 0.95, f"{share:.4f} of decision dates have obs month t-1; failing:\n{failing.to_string()}"


def test_missing_release_after_valid_gives_nan() -> None:
    releases = pd.DataFrame(
        {
            "date": [pd.Timestamp("2000-01-01")] * 3,
            "realtime_start": [pd.Timestamp("2000-01-15"), pd.Timestamp("2000-03-15"), pd.Timestamp("2000-05-15")],
            "value": [10.0, np.nan, 12.0],
        }
    )
    out = asof_from_releases(releases, pd.date_range("2000-01-31", "2000-05-31", freq="ME"))
    got = out[out["obs_month"] == pd.Timestamp("2000-01-31")].set_index("decision_date")["value"]
    assert got.loc[pd.Timestamp("2000-01-31")] == 10.0
    assert got.loc[pd.Timestamp("2000-02-29")] == 10.0
    assert np.isnan(got.loc[pd.Timestamp("2000-03-31")])
    assert np.isnan(got.loc[pd.Timestamp("2000-04-30")])
    assert got.loc[pd.Timestamp("2000-05-31")] == 12.0
