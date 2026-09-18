"""Kenneth French factor download, parse, checksum and snapshot diff.

``pull_french`` downloads the 5-factor and momentum zips, stores the zip bytes
under ``data/raw/french`` (never overwritten), parses the monthly block of
each CSV, joins them into ``Mkt-RF, SMB, HML, RMW, CMA, UMD, RF`` as decimals
and writes ``factors_<pull_id>.parquet`` through ``write_raw``. The strategy
always uses the first recorded snapshot (``french.pull_id`` in config);
``diff_french`` reports how any later pull differs from it. ``sample_end`` is
the last month in the momentum file, the binding constraint on the sample
(convention 10).
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from regime.config import Config, load_config
from regime.data.fred import append_manifest_row, new_pull_id, raw_path, write_raw

FRENCH_COLUMNS = ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD", "RF")


def parse_french_csv(text: str) -> pd.DataFrame:
    """Parse the monthly block of a French Data Library CSV.

    Header lines are skipped until the first line that starts with a comma
    followed by the column names; rows keyed ``YYYYMM`` are read until the
    first blank line; values are divided by 100; the index is the month-end
    ``Timestamp`` named ``date``.
    """
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith(","):
            start = i
            break
    if start is None:
        raise ValueError("no header line starting with a comma found")
    columns = [c.strip() for c in lines[start].split(",")[1:]]

    records: list[list] = []
    for line in lines[start + 1 :]:
        if line.strip() == "":
            break
        parts = [p.strip() for p in line.split(",")]
        key = parts[0]
        if len(key) != 6 or not key.isdigit():
            raise ValueError(f"expected a YYYYMM row key before the first blank line, got {key!r}")
        records.append([key] + [float(p) for p in parts[1:]])

    frame = pd.DataFrame(records, columns=["yyyymm"] + columns)
    frame["date"] = pd.to_datetime(frame["yyyymm"], format="%Y%m") + pd.offsets.MonthEnd(0)
    frame = frame.drop(columns="yyyymm").set_index("date")
    frame.index.name = "date"
    return frame / 100.0


def _csv_text_from_zip(zip_bytes: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"expected exactly one CSV in the zip, found {names}")
        return zf.read(names[0]).decode("utf-8", errors="replace")


def _zip_path(kind: str, pull_id: str, cfg: Config) -> Path:
    return Path(cfg.fred_raw_dir) / "french" / f"{kind}_{pull_id}.zip"


def _download(url: str) -> bytes:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return response.content


def _save_zip(path: Path, content: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"raw file already exists and is never overwritten: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def join_french(ff5: pd.DataFrame, mom: pd.DataFrame) -> pd.DataFrame:
    """Inner join on date into the seven configured columns; ``UMD`` is the momentum file's ``Mom``."""
    mom_col = [c for c in mom.columns if c.lower() == "mom"]
    if len(mom_col) != 1:
        raise ValueError(f"momentum file has no single Mom column: {list(mom.columns)}")
    joined = ff5.join(mom[[mom_col[0]]].rename(columns={mom_col[0]: "UMD"}), how="inner")
    return joined[list(FRENCH_COLUMNS)]


def pull_french(cfg: Config) -> str:
    """Download, store, parse and write the French snapshot; returns the pull_id."""
    pulled_at = datetime.now(timezone.utc)
    pull_id = new_pull_id(pulled_at)

    ff5_bytes = _download(cfg.french_factors_url)
    mom_bytes = _download(cfg.french_momentum_url)
    _save_zip(_zip_path("ff5", pull_id, cfg), ff5_bytes)
    _save_zip(_zip_path("mom", pull_id, cfg), mom_bytes)

    ff5 = parse_french_csv(_csv_text_from_zip(ff5_bytes))
    mom = parse_french_csv(_csv_text_from_zip(mom_bytes))
    factors = join_french(ff5, mom)

    raw = factors.reset_index()
    raw["pulled_at"] = pulled_at.isoformat()
    write_raw(raw, "french", "factors", pull_id, cfg)

    for series, content, parsed in (("ff5_zip", ff5_bytes, ff5), ("mom_zip", mom_bytes, mom)):
        append_manifest_row(
            cfg,
            "french",
            series,
            pull_id,
            len(parsed),
            parsed.index.min().date().isoformat(),
            parsed.index.max().date().isoformat(),
            hashlib.sha256(content).hexdigest(),
            pulled_at.isoformat(),
        )
    return pull_id


def load_french(pull_id: str, cfg: Config | None = None) -> pd.DataFrame:
    """Read ``data/raw/french/factors_<pull_id>.parquet``, indexed by month-end ``date``."""
    cfg = cfg if cfg is not None else load_config()
    frame = pd.read_parquet(raw_path("french", "factors", pull_id, cfg))
    frame = frame.drop(columns="pulled_at").set_index("date")
    frame.index = pd.DatetimeIndex(frame.index, name="date")
    return frame[list(FRENCH_COLUMNS)]


def momentum_frame(pull_id: str, cfg: Config | None = None) -> pd.DataFrame:
    """Parse the stored momentum zip of a pull."""
    cfg = cfg if cfg is not None else load_config()
    return parse_french_csv(_csv_text_from_zip(_zip_path("mom", pull_id, cfg).read_bytes()))


def sample_end(pull_id: str, cfg: Config | None = None) -> pd.Timestamp:
    """The last month in the momentum file of the pull: the sample end (convention 10)."""
    return momentum_frame(pull_id, cfg).index.max()


def diff_french(new_pull_id: str, first_pull_id: str, cfg: Config | None = None) -> pd.DataFrame:
    """Every (date, column, first_value, new_value) where two snapshots differ over their common dates."""
    first = load_french(first_pull_id, cfg)
    new = load_french(new_pull_id, cfg)
    common = first.index.intersection(new.index)
    rows = []
    for col in FRENCH_COLUMNS:
        a, b = first.loc[common, col], new.loc[common, col]
        changed = ~((a == b) | (a.isna() & b.isna()))
        for date in common[changed.to_numpy()]:
            rows.append({"date": date, "column": col, "first_value": a[date], "new_value": b[date]})
    out = pd.DataFrame(rows, columns=["date", "column", "first_value", "new_value"])
    return out.sort_values(["date", "column"]).reset_index(drop=True) if len(out) else out
