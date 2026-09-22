"""FredClient: one pull session per client, raw parquet writes under data/raw, manifest rows.

One ``FredClient`` is one pull session: ``pull_id`` is stamped once at
construction and every series pulled through the client shares it. Raw files
are never overwritten (rule 6): ``write_raw`` raises ``FileExistsError`` and
only ever appends to the manifest. ``pull_market`` and ``pull_vintages`` are
built in steps 1.3 and 1.4; ``month_end_market`` applies the 10-day lookback rule.
"""

from __future__ import annotations

import csv
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fredapi import Fred

from regime.config import Config

MANIFEST_COLUMNS = ("source", "series", "pull_id", "rows", "first_date", "last_date", "sha256", "pulled_at")
SOURCES = ("fred", "alfred", "french")
PULL_ID_FORMAT = "%Y%m%dT%H%M%SZ"


def new_pull_id(now: datetime | None = None) -> str:
    """``YYYYMMDDTHHMMSSZ`` in UTC, the key of one pull session."""
    now = now if now is not None else datetime.now(timezone.utc)
    return now.astimezone(timezone.utc).strftime(PULL_ID_FORMAT)


class FredClient:
    """One FRED/ALFRED pull session. Construction requires the API key in the environment."""

    def __init__(self, cfg: Config) -> None:
        key = os.environ.get(cfg.fred_api_key_env, "")
        if not key:
            raise RuntimeError("FRED_API_KEY not set")
        self.cfg = cfg
        self.fred = Fred(api_key=key)
        self.pulled_at = datetime.now(timezone.utc)
        self.pull_id = new_pull_id(self.pulled_at)

    def pull_market(self, series_id: str) -> str:
        """Full daily history of a market series; FRED's missing markers as NaN; returns the session pull_id."""
        raw = self.fred.get_series(series_id)
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(raw.index),
                "value": pd.to_numeric(raw.to_numpy(), errors="coerce").astype("float64"),
                "pulled_at": self.pulled_at.isoformat(),
            }
        ).sort_values("date").reset_index(drop=True)
        write_raw(frame, "fred", series_id, self.pull_id, self.cfg)
        return self.pull_id

    def pull_vintages(self, series_id: str) -> str:
        """Every release of every observation of a revised series (ALFRED), stored raw; returns the session pull_id.

        ``date`` is the observation month as FRED reports it (month start),
        ``realtime_start`` the release date. Rows are stored as returned; the
        as-of logic lives in ``regime.data.alfred``.
        """
        raw = self.fred.get_series_all_releases(series_id)
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(raw["date"]),
                "realtime_start": pd.to_datetime(raw["realtime_start"]),
                "value": pd.to_numeric(raw["value"], errors="coerce").astype("float64"),
                "pulled_at": self.pulled_at.isoformat(),
            }
        ).sort_values(["date", "realtime_start"]).reset_index(drop=True)
        write_raw(frame, "alfred", series_id, self.pull_id, self.cfg)
        return self.pull_id


def raw_path(source: str, series: str, pull_id: str, cfg: Config) -> Path:
    """``data/raw/<source>/<series>_<pull_id>.parquet``."""
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    return Path(cfg.fred_raw_dir) / source / f"{series}_{pull_id}.parquet"


def append_manifest_row(
    cfg: Config,
    source: str,
    series: str,
    pull_id: str,
    rows: int,
    first_date: str,
    last_date: str,
    sha256: str,
    pulled_at: str,
) -> None:
    """Append one row to ``cfg.fred_manifest``, writing the header first if the file is new."""
    manifest = Path(cfg.fred_manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    write_header = not manifest.exists() or manifest.stat().st_size == 0
    with manifest.open("a", newline="") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(MANIFEST_COLUMNS)
        writer.writerow([source, series, pull_id, rows, first_date, last_date, sha256, pulled_at])


def write_raw(
    frame: pd.DataFrame,
    source: str,
    series: str,
    pull_id: str,
    cfg: Config,
    sha256_of: bytes | None = None,
) -> Path:
    """Write a raw frame as parquet and append its manifest row. Never overwrites.

    ``sha256`` in the manifest is the SHA-256 of the parquet bytes unless
    ``sha256_of`` (the downloaded zip bytes, for French) is given. ``first_date``
    and ``last_date`` are the min and max of the ``date`` column (blank when
    the frame has no rows). ``pulled_at`` is the first value of the frame's
    ``pulled_at`` column so the manifest and the file carry the same timestamp.
    """
    path = raw_path(source, series, pull_id, cfg)
    if path.exists():
        raise FileExistsError(f"raw file already exists and is never overwritten: {path}")
    if "pulled_at" not in frame.columns:
        raise ValueError("raw frames carry a pulled_at column")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)

    digest = hashlib.sha256(sha256_of if sha256_of is not None else path.read_bytes()).hexdigest()
    if "date" in frame.columns and len(frame):
        dates = pd.to_datetime(frame["date"])
        first_date, last_date = dates.min().date().isoformat(), dates.max().date().isoformat()
    else:
        first_date = last_date = ""
    pulled_at = str(frame["pulled_at"].iloc[0]) if len(frame) else ""
    append_manifest_row(cfg, source, series, pull_id, len(frame), first_date, last_date, digest, pulled_at)
    return path


def month_end_from_daily(daily: pd.Series, start: pd.Timestamp, lookback_days: int) -> pd.Series:
    """Month-end values of a daily series under the lookback rule (CLAUDE.md section 3).

    For every month-end t from ``start`` to the last month-end on or before the
    last observation date, the value is the last non-missing observation with
    ``t - lookback_days days <= date <= t``, else NaN. Observations after t are
    never used.
    """
    daily = daily.dropna().sort_index()
    daily.index = pd.DatetimeIndex(daily.index)
    start = pd.Timestamp(start)
    last = daily.index.max() if len(daily) else start
    ends = pd.date_range(start, last, freq="ME", name="date")
    values = []
    for t in ends:
        window = daily.loc[(daily.index >= t - pd.Timedelta(days=lookback_days)) & (daily.index <= t)]
        values.append(window.iloc[-1] if len(window) else float("nan"))
    return pd.Series(values, index=ends, name=daily.name, dtype="float64")


def month_end_market(series_id: str, pull_id: str, cfg: Config) -> pd.Series:
    """Month-end series of ``series_id`` from the pinned raw file, named ``series_id`` and indexed by ``date``."""
    frame = pd.read_parquet(raw_path("fred", series_id, pull_id, cfg))
    daily = pd.Series(frame["value"].to_numpy(), index=pd.DatetimeIndex(frame["date"]), name=series_id)
    out = month_end_from_daily(daily, pd.Timestamp(cfg.sample_start), cfg.fred_lookback_days)
    out.name = series_id
    return out
