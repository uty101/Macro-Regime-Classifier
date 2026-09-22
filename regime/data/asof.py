"""The wide monthly as-of panel every downstream module reads (assemble_panel, build_asof_panel).

Index: month-end decision dates from ``sample.start`` to ``sample.end``, named
``date``. Columns, in order: the seven month-end market series, then for each
revised series the latest observation in the as-of-t vintage, its lag from the
same vintage, and the observation month. Nothing downstream touches raw
series (CLAUDE.md section 3).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from regime.config import Config
from regime.data.alfred import asof_path, build_asof
from regime.data.fred import month_end_market

MARKET_COLUMNS = {
    "DGS10": "dgs10",
    "DGS2": "dgs2",
    "DTWEXBGS": "dtwexbgs",
    "DTWEXM": "dtwexm",
    "DCOILWTICO": "wti",
    "VIXCLS": "vix",
    "T10YIE": "t10yie",
}
REVISED = {"CPIAUCSL": "cpi", "INDPRO": "indpro", "UNRATE": "unrate"}
PANEL_COLUMNS = (
    "dgs10", "dgs2", "dtwexbgs", "dtwexm", "wti", "vix", "t10yie",
    "cpi_m", "cpi_m3", "indpro_m", "indpro_m12", "unrate_m", "unrate_m12",
    "cpi_obs_month", "indpro_obs_month", "unrate_obs_month",
)


def _revised_columns(asof: pd.DataFrame, prefix: str, lag: int, index: pd.DatetimeIndex) -> pd.DataFrame:
    """``<prefix>_m``, ``<prefix>_m<lag>`` and ``<prefix>_obs_month`` at every decision date of ``index``.

    m is the latest observation month with a non-NaN value in the as-of-t
    vintage; the lag is the value ``lag`` observation months earlier in that
    same vintage (NaN if absent there); ``obs_month`` is m (NaT if the vintage
    holds nothing).
    """
    m_col, lag_col, obs_col = f"{prefix}_m", f"{prefix}_m{lag}", f"{prefix}_obs_month"
    out = pd.DataFrame(
        {m_col: np.nan, lag_col: np.nan, obs_col: pd.NaT}, index=index
    ).astype({m_col: "float64", lag_col: "float64", obs_col: "datetime64[ns]"})
    valid = asof.dropna(subset=["value"])
    for t, vintage in valid.groupby("decision_date"):
        if t not in index:
            continue
        vintage = vintage.set_index("obs_month")["value"].sort_index()
        m = vintage.index.max()
        m_lag = m - pd.offsets.MonthEnd(lag)
        out.loc[t, m_col] = vintage.loc[m]
        out.loc[t, lag_col] = vintage.loc[m_lag] if m_lag in vintage.index else np.nan
        out.loc[t, obs_col] = m
    return out


def assemble_panel(market: dict[str, pd.Series], asof: dict[str, pd.DataFrame], cfg: Config) -> pd.DataFrame:
    """Pure assembly of the panel from month-end market series and long as-of tables.

    ``market`` is keyed by FRED series id (the seven in ``MARKET_COLUMNS``);
    ``asof`` by revised series id (the three in ``REVISED``), each a long
    ``(decision_date, obs_month, value)`` table.
    """
    index = pd.date_range(pd.Timestamp(cfg.sample_start), pd.Timestamp(cfg.sample_end), freq="ME", name="date")
    panel = pd.DataFrame(index=index)
    for series_id, column in MARKET_COLUMNS.items():
        panel[column] = market[series_id].reindex(index).astype("float64")

    lags = {"CPIAUCSL": cfg.features_cpi_lag, "INDPRO": cfg.features_change_lag, "UNRATE": cfg.features_change_lag}
    revised = {sid: _revised_columns(asof[sid], REVISED[sid], lags[sid], index) for sid in REVISED}
    for sid, prefix in REVISED.items():
        panel[f"{prefix}_m"] = revised[sid][f"{prefix}_m"]
        panel[f"{prefix}_m{lags[sid]}"] = revised[sid][f"{prefix}_m{lags[sid]}"]
    for sid, prefix in REVISED.items():
        panel[f"{prefix}_obs_month"] = revised[sid][f"{prefix}_obs_month"]
    panel = panel[list(PANEL_COLUMNS)]
    panel.index.name = "date"
    return panel


def build_asof_panel(cfg: Config) -> pd.DataFrame:
    """Load the pinned raw and interim files, assemble the panel and write ``outputs.asof_panel``."""
    market = {sid: month_end_market(sid, cfg.fred_market_pull_id, cfg) for sid in MARKET_COLUMNS}
    asof = {}
    for sid in REVISED:
        path = asof_path(sid, cfg.fred_vintage_pull_id, cfg)
        asof[sid] = pd.read_parquet(path) if path.exists() else build_asof(sid, cfg.fred_vintage_pull_id, cfg)
    panel = assemble_panel(market, asof, cfg)
    out = Path(cfg.outputs_asof_panel)
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out)
    return panel
