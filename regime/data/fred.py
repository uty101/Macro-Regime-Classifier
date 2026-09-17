"""FredClient: one pull session per client, raw parquet writes under data/raw, manifest rows.

One ``FredClient`` is one pull session: ``pull_id`` is stamped once at
construction and every series pulled through the client shares it. Raw files
are never overwritten (rule 6): ``write_raw`` raises ``FileExistsError`` and
only ever appends to the manifest. ``pull_market`` and ``pull_vintages`` are
built in steps 1.3 and 1.4.
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
