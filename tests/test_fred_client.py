"""Step 1.2: FredClient construction and the raw-write / manifest conventions."""

import dataclasses
import hashlib
import re
from datetime import datetime, timezone

import pandas as pd
import pytest

from regime.config import load_config
from regime.data.fred import FredClient, write_raw


def _frame(n: int, start: str = "2000-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "value": [float(i) for i in range(n)],
            "pulled_at": datetime(2026, 9, 17, tzinfo=timezone.utc).isoformat(),
        }
    )


@pytest.fixture
def cfg_tmp(tmp_path):
    cfg = load_config()
    return dataclasses.replace(
        cfg,
        fred_raw_dir=str(tmp_path / "raw"),
        fred_manifest=str(tmp_path / "raw" / "manifest.csv"),
    )


def test_missing_api_key_raises(monkeypatch) -> None:
    cfg = load_config()
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with pytest.raises(RuntimeError) as excinfo:
        FredClient(cfg)
    assert str(excinfo.value) == "FRED_API_KEY not set"

    monkeypatch.setenv("FRED_API_KEY", "")
    with pytest.raises(RuntimeError) as excinfo:
        FredClient(cfg)
    assert str(excinfo.value) == "FRED_API_KEY not set"


def test_pull_id_format(monkeypatch) -> None:
    cfg = load_config()
    monkeypatch.setenv("FRED_API_KEY", "dummy-key-for-tests")
    client = FredClient(cfg)
    assert re.fullmatch(r"^\d{8}T\d{6}Z$", client.pull_id), client.pull_id


def test_write_raw_never_overwrites(cfg_tmp) -> None:
    path = write_raw(_frame(3), "fred", "DGS10", "20260917T120000Z", cfg_tmp)
    first_bytes = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_raw(_frame(5), "fred", "DGS10", "20260917T120000Z", cfg_tmp)
    assert path.read_bytes() == first_bytes


def test_manifest_appends(cfg_tmp) -> None:
    p1 = write_raw(_frame(3), "fred", "DGS10", "20260917T120000Z", cfg_tmp)
    manifest_path = cfg_tmp.fred_manifest
    with open(manifest_path, newline="") as fh:
        after_first = fh.read().splitlines()
    assert len(after_first) == 2  # header + one row

    p2 = write_raw(_frame(4, "2001-01-01"), "alfred", "CPIAUCSL", "20260917T120000Z", cfg_tmp)
    with open(manifest_path, newline="") as fh:
        after_second = fh.read().splitlines()
    assert len(after_second) == 3
    assert after_second[0] == "source,series,pull_id,rows,first_date,last_date,sha256,pulled_at"
    assert after_second[1] == after_first[1]  # the first row is byte-identical after the second write

    m = pd.read_csv(manifest_path, dtype=str)
    assert len(m) == 2
    assert m.loc[0, "sha256"] == hashlib.sha256(p1.read_bytes()).hexdigest()
    assert m.loc[1, "sha256"] == hashlib.sha256(p2.read_bytes()).hexdigest()
    assert list(m["rows"]) == ["3", "4"]
    assert list(m["first_date"]) == ["2000-01-01", "2001-01-01"]
    assert list(m["last_date"]) == ["2000-01-03", "2001-01-04"]
    assert list(m["source"]) == ["fred", "alfred"]


def test_sha256_of_overrides_parquet_digest(cfg_tmp) -> None:
    zip_bytes = b"not really a zip"
    write_raw(_frame(2), "french", "factors", "20260917T120000Z", cfg_tmp, sha256_of=zip_bytes)
    m = pd.read_csv(cfg_tmp.fred_manifest, dtype=str)
    assert m.loc[0, "sha256"] == hashlib.sha256(zip_bytes).hexdigest()
