"""Step 1.1: config.toml and Config are the same set of keys, and Config is frozen and typed."""

import dataclasses
import re
import tomllib
from pathlib import Path

import pytest

from regime.config import Config, load_config

ROOT = Path(__file__).resolve().parents[1]


def _toml_keys() -> set[str]:
    with (ROOT / "config.toml").open("rb") as fh:
        raw = tomllib.load(fh)
    return {f"{table}_{key}" for table, keys in raw.items() for key in keys}


def test_config_keys_round_trip() -> None:
    toml_keys = _toml_keys()
    field_names = {f.name for f in dataclasses.fields(Config)}
    only_in_toml = sorted(toml_keys - field_names)
    only_in_config = sorted(field_names - toml_keys)
    assert toml_keys == field_names, (
        f"symmetric difference — only in config.toml: {only_in_toml}; "
        f"only in Config: {only_in_config}"
    )


def test_config_is_frozen_and_typed() -> None:
    cfg = load_config(str(ROOT / "config.toml"))
    assert isinstance(cfg, Config)

    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.run_seed = 0  # type: ignore[misc]

    assert cfg.run_seed == 20260917

    with (ROOT / "config.toml").open("rb") as fh:
        raw = tomllib.load(fh)
    for table, keys in raw.items():
        for key, value in keys.items():
            if isinstance(value, list):
                got = getattr(cfg, f"{table}_{key}")
                assert isinstance(got, tuple), f"{table}.{key} should be a tuple, got {type(got)}"
                assert got == tuple(value)

    for f in dataclasses.fields(Config):
        assert isinstance(getattr(cfg, f.name), (int, float, str, bool, tuple)), f.name

    # Two-phase behaviour. From step 1.1 on, every *_pull_id field and sample_end
    # are str (they start as "" placeholders). The step that fills a key tightens
    # the assertion for that key to non-empty in the same commit:
    #   step 1.3 -> fred_market_pull_id, step 1.4 -> fred_vintage_pull_id,
    #   step 1.5 -> french_pull_id and sample_end.
    pull_id_fields = [f.name for f in dataclasses.fields(Config) if f.name.endswith("_pull_id")]
    assert pull_id_fields == ["fred_market_pull_id", "fred_vintage_pull_id", "french_pull_id"]
    for name in pull_id_fields + ["sample_end"]:
        assert isinstance(getattr(cfg, name), str), name
    # Filled at step 1.5 (French pull 20260918T083009Z, momentum file ends 2026-07):
    assert re.fullmatch(r"\d{8}T\d{6}Z", cfg.french_pull_id), cfg.french_pull_id
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", cfg.sample_end), cfg.sample_end
    assert cfg.sample_end != ""
