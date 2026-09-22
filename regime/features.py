"""Raw feature construction, dollar splice and real-time standardisation (section 2).

Everything here reads the wide as-of panel of ``regime/data/asof.py`` and
nothing else. Lags are rows of the decision-date index (convention 11): the
market series use ``shift(cfg.features_change_lag)`` on the panel, and the
revised series use the ``_m3`` / ``_m12`` columns the panel already carries
from the same as-of vintage (convention 12). Logs of a non-positive value are
NaN, never -inf, and the row is dropped by ``model_input`` rather than patched.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from regime.config import Config


def _log(s: pd.Series) -> pd.Series:
    """Natural log with non-positive values mapped to NaN instead of -inf or a warning."""
    return np.log(s.where(s > 0))


def splice_dollar_chg12(panel: pd.DataFrame, cfg: Config) -> pd.Series:
    """12-row log change of the dollar index: DTWEXM before the splice date, DTWEXBGS from it.

    Changes are spliced, never levels, so the two bases never meet. DTWEXBGS
    begins in 2006-01, so its first 12-row change is valid at 2007-01-31, the
    splice date row.
    """
    lag = cfg.features_change_lag
    m = _log(panel["dtwexm"] / panel["dtwexm"].shift(lag)) * 100
    b = _log(panel["dtwexbgs"] / panel["dtwexbgs"].shift(lag)) * 100
    splice = pd.Timestamp(cfg.features_dollar_splice_date)
    out = m.where(panel.index < splice, b)
    out.name = "dollar_chg12"
    return out


def build_raw_features(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """The ten raw features on the panel's index: the eight core columns then the two robustness columns.

    Written to ``cfg.outputs_features_raw``.
    """
    lag = cfg.features_change_lag
    power = 12 / cfg.features_cpi_lag
    raw = pd.DataFrame(index=panel.index)
    raw["dgs10_level"] = panel["dgs10"]
    raw["dgs10_chg12"] = panel["dgs10"] - panel["dgs10"].shift(lag)
    raw["slope_2s10s"] = panel["dgs10"] - panel["dgs2"]
    raw["cpi_3m_ann"] = ((panel["cpi_m"] / panel["cpi_m3"]) ** power - 1) * 100
    raw["indpro_chg12"] = _log(panel["indpro_m"] / panel["indpro_m12"]) * 100
    raw["dollar_chg12"] = splice_dollar_chg12(panel, cfg)
    raw["oil_chg12"] = _log(panel["wti"] / panel["wti"].shift(lag)) * 100
    raw["log_vix"] = _log(panel["vix"])
    raw["unrate_chg12"] = panel["unrate_m"] - panel["unrate_m12"]
    raw["breakeven_chg12"] = panel["t10yie"] - panel["t10yie"].shift(lag)
    raw = raw[list(cfg.features_core) + list(cfg.features_robustness)].astype("float64")
    raw.index.name = "date"
    _write(raw, cfg.outputs_features_raw)
    return raw


def standardise(raw: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Real-time z-scores: first-window moments up to ``first_window_end``, expanding moments after it.

    Rows from ``cfg.sample_features_from``. For t <= ``first_window_end`` every
    column uses the mean and standard deviation (ddof = ``cfg.features_ddof``)
    of its non-NaN rows in ``[features_from, first_window_end]`` (convention 2).
    For t after it the moments are over ``[features_from, t]`` inclusive. NaN
    rows of a column contribute nothing to its moments and stay NaN in z, so
    the robustness columns follow the same rule over their own non-NaN rows.
    Written once to ``cfg.outputs_features_z`` (convention 3).
    """
    start = pd.Timestamp(cfg.sample_features_from)
    window_end = pd.Timestamp(cfg.sample_first_window_end)
    ddof = cfg.features_ddof
    x = raw.loc[raw.index >= start].astype("float64")

    window = x.loc[x.index <= window_end]
    mu = x.expanding().mean()
    sd = x.expanding().std(ddof=ddof)
    in_window = x.index <= window_end
    mu.loc[in_window] = np.broadcast_to(window.mean().to_numpy(), (int(in_window.sum()), x.shape[1]))
    sd.loc[in_window] = np.broadcast_to(window.std(ddof=ddof).to_numpy(), (int(in_window.sum()), x.shape[1]))

    z = (x - mu) / sd
    z.index.name = "date"
    _write(z, cfg.outputs_features_z)
    return z


def model_input(z: pd.DataFrame, cfg: Config, columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    """The complete-case model matrix: ``columns`` (default the core eight), dates >= ``features_from``, no NaN.

    The dropped dates and their NaN columns are written to
    ``cfg.outputs_dropped_rows`` with columns ``date, missing``; ``missing`` is
    the NaN column names joined by ``;``.
    """
    cols = list(columns if columns is not None else cfg.features_core)
    start = pd.Timestamp(cfg.sample_features_from)
    sub = z.loc[z.index >= start, cols]
    nan_mask = sub.isna()
    dropped_dates = sub.index[nan_mask.any(axis=1)]
    dropped = pd.DataFrame(
        {
            "date": dropped_dates,
            "missing": [";".join(nan_mask.columns[nan_mask.loc[d].to_numpy()]) for d in dropped_dates],
        }
    )
    path = Path(cfg.outputs_dropped_rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    dropped.to_csv(path, index=False, date_format="%Y-%m-%d")
    out = sub.loc[~nan_mask.any(axis=1)]
    out.index.name = "date"
    return out


def _write(frame: pd.DataFrame, path_str: str) -> None:
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
