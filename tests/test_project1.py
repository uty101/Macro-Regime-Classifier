"""Step 1.6: the optional project 1 adapter."""

import logging

import pandas as pd

from regime.data.project1 import load_project1


def test_absent_returns_empty_schema_and_logs(tmp_path, caplog) -> None:
    with caplog.at_level(logging.INFO, logger="regime"):
        frame = load_project1(str(tmp_path / "does_not_exist.parquet"))
    assert "project1: absent" in caplog.text
    assert list(frame.columns) == ["date", "factor", "ret"]
    assert len(frame) == 0
    assert str(frame["date"].dtype) == "datetime64[ns]"
    assert frame["ret"].dtype == "float64"


def test_present_round_trip(tmp_path) -> None:
    src = pd.DataFrame(
        {
            "date": [pd.Timestamp("2010-01-31"), pd.Timestamp("2010-02-28")],
            "factor": ["value", "value"],
            "ret": [0.0123, -0.0045],
        }
    )
    path = tmp_path / "project1_factors.parquet"
    src.to_parquet(path, index=False)
    got = load_project1(str(path))
    pd.testing.assert_frame_equal(got, src)
