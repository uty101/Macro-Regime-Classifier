"""Config: a frozen dataclass with one field per config.toml key, named <table>_<key>.

This is the only module that reads config.toml. Every other function takes
``cfg: Config``. TOML arrays become tuples; dates stay ISO strings and are
converted with ``pd.Timestamp(...)`` where used. ``load_config`` raises
``KeyError`` naming the offending key when the TOML and the dataclass disagree,
in either direction, so the two can never drift silently.
"""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # [run]
    run_seed: int
    run_step_timeout_minutes: int
    run_long_step_timeout_minutes: int
    run_long_steps: tuple

    # [sample]
    sample_start: str
    sample_end: str
    sample_features_from: str
    sample_first_window_end: str
    sample_robustness_from: str

    # [fred]
    fred_api_key_env: str
    fred_market_series: tuple
    fred_vintage_series: tuple
    fred_lookback_days: int
    fred_raw_dir: str
    fred_manifest: str
    fred_market_pull_id: str
    fred_vintage_pull_id: str

    # [french]
    french_factors_url: str
    french_momentum_url: str
    french_columns: tuple
    french_pull_id: str

    # [features]
    features_core: tuple
    features_core_no_level: tuple
    features_primary: str
    features_robustness: tuple
    features_change_lag: int
    features_cpi_lag: int
    features_dollar_splice_date: str
    features_ddof: int

    # [hmm]
    hmm_covariance_type: str
    hmm_n_iter: int
    hmm_tol: float
    hmm_min_covar: float
    hmm_init_params: str
    hmm_params: str
    hmm_n_restarts: int
    hmm_k_grid: tuple
    hmm_k_candidates: tuple
    hmm_refit_month: int
    hmm_min_state_rows: int
    hmm_assigned_threshold: float
    hmm_anchor_feature: str
    hmm_anchor_tiebreak_feature: str
    hmm_anchor_tie_tolerance: float
    hmm_robustness_first_refit: str
    hmm_robustness_covariance_type: str

    # [gmm]
    gmm_covariance_type: str
    gmm_n_init: int
    gmm_max_iter: int
    gmm_tol: float
    gmm_reg_covar: float
    gmm_n_restarts: int

    # [rules]
    rules_growth_feature: str
    rules_inflation_feature: str

    # [bootstrap]
    bootstrap_block_size: int
    bootstrap_n_replications: int
    bootstrap_p_low: float
    bootstrap_p_high: float
    bootstrap_block_size_grid: tuple

    # [strategy]
    strategy_factors: tuple
    strategy_universe: tuple
    strategy_min_regime_obs: int
    strategy_min_regime_obs_grid: tuple
    strategy_eta_grid: tuple
    strategy_lag_grid: tuple
    strategy_cost_bp_grid: tuple
    strategy_headline_eta: float
    strategy_headline_lag: int
    strategy_headline_cost_bp: int
    strategy_headline_source: str
    strategy_label_sources: tuple

    # [outputs]
    outputs_interim_dir: str
    outputs_processed_dir: str
    outputs_external_dir: str
    outputs_tables_dir: str
    outputs_charts_dir: str
    outputs_regimes_dir: str
    outputs_chart_dpi: int
    outputs_asof_panel: str
    outputs_features_raw: str
    outputs_features_z: str
    outputs_dropped_rows: str
    outputs_primary_k: str
    outputs_project1_file: str
    outputs_filtered_probs: str
    outputs_smoothed_probs: str
    outputs_gmm_filtered_probs: str
    outputs_regime_labels: str
    outputs_timing_results: str
    outputs_runtime_log: str


_ALLOWED_TYPES = (int, float, str, bool, tuple)

FEATURE_SETS = ("core", "core_no_level")


def primary_columns(cfg: "Config") -> tuple:
    """The model-input columns named by ``cfg.features_primary``.

    Every ``model_input`` call from section 3 onward resolves its columns
    through here, so switching the primary feature set is a one-key change in
    ``config.toml`` and never a change in the pipeline code. The non-primary
    set is what step 6.6 reruns.
    """
    return feature_set_columns(cfg, cfg.features_primary)


def feature_set_columns(cfg: "Config", name: str) -> tuple:
    """The columns of a named feature set: ``"core"`` (d = 8) or ``"core_no_level"`` (d = 7)."""
    if name == "core":
        return tuple(cfg.features_core)
    if name == "core_no_level":
        return tuple(cfg.features_core_no_level)
    raise ValueError(f"unknown feature set {name!r}; expected one of {FEATURE_SETS}")


def flatten_toml(raw: dict) -> dict:
    """``{table: {key: value}}`` -> ``{f"{table}_{key}": value}`` with lists as tuples."""
    flat: dict = {}
    for table, keys in raw.items():
        if not isinstance(keys, dict):
            raise KeyError(f"top-level key {table!r} is not a table")
        for key, value in keys.items():
            flat[f"{table}_{key}"] = tuple(value) if isinstance(value, list) else value
    return flat


def load_config(path: str = "config.toml") -> Config:
    """Read ``config.toml`` into a ``Config``.

    Raises ``KeyError`` naming the first TOML key with no dataclass field, or
    the first field with no TOML key.
    """
    with Path(path).open("rb") as fh:
        raw = tomllib.load(fh)
    flat = flatten_toml(raw)

    fields = {f.name: f for f in dataclasses.fields(Config)}
    for name in sorted(flat):
        if name not in fields:
            raise KeyError(f"config.toml key {name!r} has no Config field")
    for name in fields:
        if name not in flat:
            raise KeyError(f"Config field {name!r} has no config.toml key")

    values: dict = {}
    for name, field in fields.items():
        value = flat[name]
        # TOML writes 500 and 1e-4 differently; a float field accepts an int value.
        if field.type in ("float", float) and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        if not isinstance(value, _ALLOWED_TYPES):
            raise TypeError(f"config.toml key {name!r} has unsupported type {type(value).__name__}")
        values[name] = value
    return Config(**values)
