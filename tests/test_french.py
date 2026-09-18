"""Step 1.5: French CSV parsing, the pinned 2010-01 row and the sample end."""

import pandas as pd

from regime.config import load_config
from regime.data.french import load_french, parse_french_csv, sample_end

SYNTHETIC = """This file was created using the 202607 CRSP database.
Some more header text

,Mkt-RF,SMB,HML,RMW,CMA,RF
201001,   -3.35,    0.40,    0.33,   -1.08,    0.50,    0.00
201002,    3.40,    1.19,    3.15,   -0.29,    1.22,    0.00
201003,    6.31,    1.47,    2.13,   -0.62,    1.47,    0.01

 Annual Factors: January-December 
,Mkt-RF,SMB,HML,RMW,CMA,RF
2010,   17.37,   13.58,   -5.32,    7.51,   -3.44,    0.12
2011,    0.44,   -6.05,   -8.41,    9.83,    2.14,    0.04
"""


def test_parse_stops_at_first_blank_line() -> None:
    frame = parse_french_csv(SYNTHETIC)
    assert len(frame) == 3
    assert list(frame.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    assert list(frame.index) == [
        pd.Timestamp("2010-01-31"),
        pd.Timestamp("2010-02-28"),
        pd.Timestamp("2010-03-31"),
    ]
    assert frame.index.name == "date"
    assert frame.index.is_month_end.all()
    assert abs(frame.loc["2010-01-31", "Mkt-RF"] - (-0.0335)) < 1e-12
    assert abs(frame.loc["2010-03-31", "RF"] - 0.0001) < 1e-12
    assert 2010 not in [d.year for d in frame.index if d.month == 12]  # the annual block is not read


def test_2010_01_row_matches_site() -> None:
    # Read by eye from the downloaded CSVs of pull 20260918T083009Z:
    #   F-F_Research_Data_5_Factors_2x3.csv: "201001,   -3.35,    0.40,    0.33,   -1.08,    0.50,    0.00"
    #   F-F_Momentum_Factor.csv:             "201001,  -5.31"
    cfg = load_config()
    row = load_french(cfg.french_pull_id).loc[pd.Timestamp("2010-01-31")]
    expected = {
        "Mkt-RF": -3.35 / 100,
        "SMB": 0.40 / 100,
        "HML": 0.33 / 100,
        "RMW": -1.08 / 100,
        "CMA": 0.50 / 100,
        "UMD": -5.31 / 100,
        "RF": 0.00 / 100,
    }
    assert list(row.index) == list(expected)
    for col, value in expected.items():
        assert abs(row[col] - value) < 1e-12, (col, row[col], value)


def test_sample_end_is_momentum_last_month() -> None:
    cfg = load_config()
    assert sample_end(cfg.french_pull_id) == pd.Timestamp(cfg.sample_end)
    assert pd.Timestamp(cfg.sample_end).is_month_end
