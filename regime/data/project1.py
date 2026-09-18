"""Optional project 1 factor series adapter (load_project1).

Project 1 series are optional (convention 7). ``load_project1()`` reads
``outputs.project1_file`` if it exists and returns the long frame
``(date, factor, ret)``; otherwise it logs ``project1: absent`` and returns an
empty frame with the same three columns and dtypes. Nothing may depend on the
file being present.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from regime.config import load_config

log = logging.getLogger("regime")

PROJECT1_COLUMNS = ("date", "factor", "ret")


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.Series(dtype="datetime64[ns]"),
            "factor": pd.Series(dtype="str"),
            "ret": pd.Series(dtype="float64"),
        }
    )


def load_project1(path: str | None = None) -> pd.DataFrame:
    """Columns ``date`` (month-end Timestamp), ``factor`` (str), ``ret`` (float); empty if the file is absent.

    ``path`` exists only so tests can point at a temporary file; the
    no-argument call reads ``load_config().outputs_project1_file``.
    """
    file = Path(path if path is not None else load_config().outputs_project1_file)
    if not file.exists():
        log.info("project1: absent")
        return _empty()
    frame = pd.read_parquet(file)
    missing = [c for c in PROJECT1_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"project1 file lacks columns {missing}: {file}")
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(frame["date"]) + pd.offsets.MonthEnd(0),
            "factor": frame["factor"].astype(str),
            "ret": frame["ret"].astype("float64"),
        }
    )
    return out.sort_values(["date", "factor"]).reset_index(drop=True)
