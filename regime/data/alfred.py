"""As-of vintage tables from ALFRED release histories (build_asof, asof_from_releases).

For decision date t, the vintage as of t holds, for each observation month m,
the value of the release with the greatest ``realtime_start <= t`` (CLAUDE.md
section 3). A month with no release on or before t is NaN. A month whose
latest release on or before t carries a missing value is NaN too: what is
forward-filled across release dates is which release is latest, never the
value itself, so an earlier valid value is never served in place of a later
missing one. A decision date before the first release of anything has all
values NaN. There is no fallback to the current vintage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from regime.config import Config
from regime.data.fred import raw_path

ASOF_COLUMNS = ("decision_date", "obs_month", "value")
_MISSING = np.inf  # stands in for a missing release value inside the forward fill


def asof_from_releases(releases: pd.DataFrame, decision_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Long as-of table ``(decision_date, obs_month, value)`` from ``(date, realtime_start, value)`` releases.

    For each decision date t and each observation month m <= t, ``value`` is
    exactly the value of the release with the greatest ``realtime_start <= t``:
    NaN if that release carries a missing value, NaN if no such release exists.
    """
    rel = releases[["date", "realtime_start", "value"]].copy()
    rel["obs_month"] = pd.to_datetime(rel["date"]) + pd.offsets.MonthEnd(0)
    rel["realtime_start"] = pd.to_datetime(rel["realtime_start"])
    rel = rel.sort_values(["obs_month", "realtime_start"]).reset_index(drop=True)

    # A missing release must still count as the latest release, so it is
    # stored as a sentinel (no real value is infinite), the forward fill then
    # carries "which release is latest" down the release dates, and the
    # sentinel is mapped back to NaN at the end.
    rel["value"] = rel["value"].astype("float64").fillna(_MISSING)

    # wide: rows = release dates (ascending), columns = observation months; each
    # cell is that release's value for that month, forward-filled down the rows
    # so that row r holds the value of the latest release on or before release date r.
    wide = rel.pivot_table(index="realtime_start", columns="obs_month", values="value", aggfunc="last", dropna=False)
    wide = wide.sort_index().ffill()
    release_dates = wide.index

    frames = []
    for t in decision_dates:
        months = wide.columns[wide.columns <= t]
        pos = release_dates.searchsorted(t, side="right") - 1
        if pos < 0 or len(months) == 0:
            values = np.full(len(months), np.nan)
        else:
            values = wide.iloc[pos][months].to_numpy(dtype="float64")
        frames.append(pd.DataFrame({"decision_date": t, "obs_month": months, "value": values}))
    if not frames:
        return pd.DataFrame(columns=list(ASOF_COLUMNS))
    out = pd.concat(frames, ignore_index=True)
    out["decision_date"] = pd.to_datetime(out["decision_date"])
    out["obs_month"] = pd.to_datetime(out["obs_month"])
    out.loc[out["value"] == _MISSING, "value"] = np.nan
    return out[list(ASOF_COLUMNS)]


def load_releases(series_id: str, pull_id: str, cfg: Config) -> pd.DataFrame:
    """The raw ALFRED release table of a pinned pull."""
    return pd.read_parquet(raw_path("alfred", series_id, pull_id, cfg))


def asof_path(series_id: str, pull_id: str, cfg: Config) -> Path:
    return Path(cfg.outputs_interim_dir) / f"asof_{series_id}_{pull_id}.parquet"


def build_asof(series_id: str, pull_id: str, cfg: Config) -> pd.DataFrame:
    """As-of table for every month-end from ``sample_start`` to the last month-end <= the last release date; written to data/interim."""
    releases = load_releases(series_id, pull_id, cfg)
    last_release = pd.to_datetime(releases["realtime_start"]).max()
    decision_dates = pd.date_range(pd.Timestamp(cfg.sample_start), last_release, freq="ME", name="decision_date")
    out = asof_from_releases(releases, decision_dates)
    path = asof_path(series_id, pull_id, cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out
