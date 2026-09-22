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


def test_conditional_join_skipped_when_absent(tmp_path, caplog):
    """Convention 7: an absent project 1 file is a logged skip, not a failure and not a file."""
    import dataclasses
    import logging

    import pandas as pd

    from regime.config import load_config
    from regime.run import project1_conditional

    cfg = dataclasses.replace(
        load_config(),
        outputs_project1_file=str(tmp_path / "project1_factors.parquet"),
        outputs_tables_dir=str(tmp_path / "tables"),
    )
    index = pd.date_range("2005-01-31", periods=6, freq="ME", name="date")
    labels = pd.DataFrame({"label": [0, 0, 1, 1, 0, 1], "assigned": [True] * 6}, index=index)

    with caplog.at_level(logging.INFO, logger="regime"):
        result = project1_conditional(cfg, labels)

    assert result is None
    assert "project1: absent, conditional join skipped" in caplog.text
    assert list(tmp_path.rglob("conditional_stats_project1.csv")) == []


def test_conditional_join_runs_when_present(tmp_path):
    """With a file present the same bootstrap runs, over the project 1 factor tuple."""
    import dataclasses

    import numpy as np
    import pandas as pd

    from regime.config import load_config
    from regime.run import project1_conditional

    cfg = dataclasses.replace(
        load_config(),
        outputs_project1_file=str(tmp_path / "project1_factors.parquet"),
        outputs_tables_dir=str(tmp_path / "tables"),
        bootstrap_n_replications=25,
        sample_first_window_end="2005-01-31",
    )
    rng = np.random.default_rng(cfg.run_seed)
    index = pd.date_range("2005-01-31", periods=40, freq="ME", name="date")
    long = pd.concat(
        [
            pd.DataFrame({"date": index, "factor": name, "ret": rng.normal(0.004, 0.02, size=len(index))})
            for name in ("VALUE", "QUALITY")
        ],
        ignore_index=True,
    )
    long.to_parquet(cfg.outputs_project1_file)
    labels = pd.DataFrame(
        {"label": list(rng.integers(0, 3, size=len(index) - 1)), "assigned": True}, index=index[:-1]
    )

    stats = project1_conditional(cfg, labels)

    assert stats is not None
    assert set(stats["factor"]) == {"VALUE", "QUALITY"}
    assert list(stats.columns) == [
        "factor", "state", "n", "ann_mean", "ann_std", "sharpe", "sharpe_p05", "sharpe_p95", "excludes_zero",
    ]
    assert (tmp_path / "tables" / "conditional_stats_project1.csv").exists()
