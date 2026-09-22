"""Entry point: ``python -m regime.run [--section N] [--pull]``.

Without ``--section`` every section 1 to 7 runs in order; with it, one
section. Each section is ``section_N(cfg, pull=False)`` registered in
``SECTIONS``; a section not yet built raises
``NotImplementedError("section N not built")``. ``--pull`` is honoured only by
section 1 and performs fresh raw pulls (new pull_id, new raw files, manifest
rows) without changing config.toml; without it section 1 rebuilds the as-of
panel from the pinned pull_ids. Every section rebuilds its outputs from
data/processed/ deterministically, so running twice produces identical files.
Wall-clock seconds per section are appended to outputs/tables/runtime.csv from
step 7.3 onward; until then the registry only.
"""

from __future__ import annotations

import argparse
import logging
from typing import Callable

from regime.config import Config, load_config


def _not_built(n: int) -> Callable[[Config, bool], None]:
    def section(cfg: Config, pull: bool = False) -> None:
        raise NotImplementedError(f"section {n} not built")

    section.__name__ = f"section_{n}"
    return section


def section_1(cfg: Config, pull: bool = False) -> None:
    """Section 1: raw pulls (with --pull) and the as-of panel from the pinned pull ids."""
    import pandas as pd

    from regime.data.alfred import build_asof, load_releases
    from regime.data.asof import build_asof_panel
    from regime.data.fred import FredClient, month_end_market
    from regime.data.french import diff_french, pull_french
    from regime.data.project1 import load_project1

    log = logging.getLogger("regime")
    if pull:
        client = FredClient(cfg)
        for series_id in cfg.fred_market_series:
            client.pull_market(series_id)
            log.info("pulled market series %s under pull_id %s", series_id, client.pull_id)
        log.info("market pull_id %s (pin as fred.market_pull_id in config.toml)", client.pull_id)

        vclient = FredClient(cfg)
        for series_id in cfg.fred_vintage_series:
            vclient.pull_vintages(series_id)
            rel = load_releases(series_id, vclient.pull_id, cfg)
            earliest = pd.to_datetime(rel["realtime_start"]).min()
            log.info(
                "pulled vintages %s under pull_id %s: %d rows, %d distinct realtime_start, earliest %s",
                series_id, vclient.pull_id, len(rel), rel["realtime_start"].nunique(), earliest.date(),
            )
            if earliest >= pd.Timestamp("1995-01-01"):
                log.error("%s: earliest realtime_start %s is not before 1995-01-01; pull treated as truncated (OPEN.md)", series_id, earliest.date())
        log.info("vintage pull_id %s (pin as fred.vintage_pull_id in config.toml)", vclient.pull_id)

    market_pull_id = cfg.fred_market_pull_id
    if market_pull_id:
        for series_id in cfg.fred_market_series:
            s = month_end_market(series_id, market_pull_id, cfg)
            log.info("%s: %d month-ends, %d NaN", series_id, len(s), int(s.isna().sum()))
    else:
        log.info("fred.market_pull_id not pinned; no month-end series built")

    vintage_pull_id = cfg.fred_vintage_pull_id
    if vintage_pull_id:
        for series_id in cfg.fred_vintage_series:
            rel = load_releases(series_id, vintage_pull_id, cfg)
            earliest = pd.to_datetime(rel["realtime_start"]).min()
            if earliest >= pd.Timestamp("1995-01-01"):
                raise RuntimeError(f"{series_id}: earliest realtime_start {earliest.date()} is not before 1995-01-01")
            table = build_asof(series_id, vintage_pull_id, cfg)
            log.info("%s: as-of table %d rows, %d decision dates", series_id, len(table), table["decision_date"].nunique())
    else:
        log.info("fred.vintage_pull_id not pinned; no as-of tables built")

    if pull:
        new_french = pull_french(cfg)
        log.info("french pull_id %s (first snapshot pinned as french.pull_id is %s)", new_french, cfg.french_pull_id)
        if cfg.french_pull_id and new_french != cfg.french_pull_id:
            diff = diff_french(new_french, cfg.french_pull_id, cfg)
            log.info("diff_french(%s, %s): %d differing cells", new_french, cfg.french_pull_id, len(diff))
            print(diff.to_string() if len(diff) else "<no differences over common dates>")

    project1 = load_project1()
    log.info("project1: %d rows", len(project1))

    if market_pull_id and vintage_pull_id:
        panel = build_asof_panel(cfg)
        log.info("as-of panel written to %s: %d rows x %d columns", cfg.outputs_asof_panel, *panel.shape)
    else:
        log.info("as-of panel not built: market_pull_id or vintage_pull_id not pinned")


def section_2(cfg: Config, pull: bool = False) -> None:
    """Section 2: raw features, the dollar splice, real-time z-scores, the sanity table and the review chart."""
    from pathlib import Path

    import pandas as pd

    from regime.charts import features_review_chart
    from regime.features import build_raw_features, model_input, standardise
    from regime.tables import feature_sanity

    log = logging.getLogger("regime")
    panel = pd.read_parquet(cfg.outputs_asof_panel)
    log.info("as-of panel read from %s: %d rows x %d columns", cfg.outputs_asof_panel, *panel.shape)

    raw = build_raw_features(panel, cfg)                                   # 2.1 and 2.2
    log.info("raw features written to %s: %d rows x %d columns", cfg.outputs_features_raw, *raw.shape)
    z = standardise(raw, cfg)                                               # 2.3
    log.info("z written to %s: %d rows from %s", cfg.outputs_features_z, len(z), z.index[0].date())
    x = model_input(z, cfg)
    dropped = pd.read_csv(cfg.outputs_dropped_rows)
    log.info("model input %d rows x %d columns; %d dropped rows written to %s", *x.shape, len(dropped), cfg.outputs_dropped_rows)
    for row in dropped.itertuples(index=False):
        log.info("dropped %s: %s", row.date, row.missing)

    table = feature_sanity(raw, cfg)                                        # 2.4
    log.info("feature_sanity.csv written: %d rows", len(table))
    chart = Path(cfg.outputs_charts_dir) / "features_review.png"
    features_review_chart(raw, str(chart), cfg)
    log.info("features review chart written to %s", chart)


ROBUSTNESS_DIR_NAME = {"core": "level", "core_no_level": "nolevel"}


def _variant_cfg(cfg: Config, name: str):
    """``cfg`` with every output path redirected under ``robustness/<name>/``.

    The non-primary feature set writes here. Nothing in the main outputs is
    touched, and the variant is driven entirely by ``dataclasses.replace`` —
    never by editing ``config.toml`` (section 6's rule, applied early because
    section 3 now runs two feature sets).
    """
    import dataclasses

    folder = ROBUSTNESS_DIR_NAME[name]
    return dataclasses.replace(
        cfg,
        outputs_tables_dir=f"{cfg.outputs_tables_dir}/robustness/{folder}",
        outputs_filtered_probs=f"{cfg.outputs_regimes_dir}/robustness/{folder}/filtered_probs.csv",
        outputs_smoothed_probs=f"{cfg.outputs_regimes_dir}/robustness/{folder}/smoothed_probs.csv",
        outputs_gmm_filtered_probs=f"{cfg.outputs_regimes_dir}/robustness/{folder}/gmm_filtered_probs.csv",
        outputs_dropped_rows=f"{cfg.outputs_processed_dir}/dropped_rows_{folder}.csv",
        outputs_primary_k=f"{cfg.outputs_processed_dir}/primary_k_{folder}.txt",
    )


def _run_feature_set(z, columns, K: int, cfg: Config) -> dict:
    """The whole classifier pipeline for one column set: steps 3.4 to 3.7.

    State numbering is chained throughout (convention 16): the expanding HMM
    sorts its first refit and chains the rest, the smoothed fit chains to the
    last expanding refit, and the GMM chains from the expanding HMM's first
    anchored refit. So every frame this returns numbers the same regime the
    same way and the three can be compared date by date.
    """
    from pathlib import Path as _Path

    import pandas as pd

    from regime.features import model_input
    from regime.models.gmm import run_expanding_gmm
    from regime.models.hmm import hard_labels, run_expanding_hmm, run_smoothed_hmm
    from regime.tables import write_hmm_tables

    x = model_input(z, cfg, columns=tuple(columns))
    filtered, params = run_expanding_hmm(x, K, cfg)                          # 3.4
    smoothed, _ = run_smoothed_hmm(x, K, cfg, chain_to=params[-1].means)     # 3.5
    gmm = run_expanding_gmm(x, K, cfg, chain_to=params[0].means)             # 3.6
    write_hmm_tables(params, list(x.columns), cfg)                           # 3.7

    tables = _Path(cfg.outputs_tables_dir)
    return {
        "model_input": x,
        "params": params,
        "filtered_labels": hard_labels(filtered, cfg),
        "smoothed_labels": hard_labels(smoothed, cfg),
        "gmm_labels": hard_labels(gmm, cfg),
        "durations": pd.read_csv(tables / "expected_duration.csv"),
        "state_counts": pd.read_csv(tables / "state_counts.csv"),
        "chain_table": pd.read_csv(tables / "anchor_chain.csv"),
    }


def section_3(cfg: Config, pull: bool = False) -> None:
    """Section 3: the four label sources, from features_z.parquet — steps 3.1 to 3.7.

    Nothing here reads raw data or recomputes z (convention 3), and nothing
    here reads a factor return. Both feature sets are run every time: the
    primary one named by ``cfg.features_primary`` into the main outputs, the
    other into ``robustness/<name>/``. The diagnostics that decide which is
    primary are therefore regenerated on every run and can never go stale
    against the config key they justify.
    """
    from pathlib import Path as _Path

    import pandas as pd

    from regime.config import FEATURE_SETS, feature_set_columns
    from regime.models.hmm import refit_dates, select_k
    from regime.models.rules import rules_labels
    from regime.tables import (
        CLASSIFIER_DIAGNOSTIC_COLUMNS,
        classifier_diagnostics_row,
        primary_feature_set_decision,
    )

    log = logging.getLogger("regime")
    raw = pd.read_parquet(cfg.outputs_features_raw)
    z = pd.read_parquet(cfg.outputs_features_z)

    labels = rules_labels(raw, cfg)                                          # 3.1
    log.info("rules labels: %d rows, value counts %s", len(labels), labels.value_counts().sort_index().to_dict())

    primary = cfg.features_primary
    other = next(name for name in FEATURE_SETS if name != primary)
    window_end = pd.Timestamp(cfg.sample_first_window_end)

    # 3.3 — K is selected once, on the d = 8 core set, and used for both. The
    # other set's BIC table is written for information only; K is not
    # reselected, so a feature-set change cannot silently change K too.
    core_input = _model_input_for(z, cfg, "core")
    primary_K, _bic = select_k(core_input.loc[core_input.index <= window_end], cfg)
    log.info("primary_K = %d from the core (d=%d) first window", primary_K, core_input.shape[1])

    results, diagnostics = {}, []
    for name in (primary, other):
        run_cfg = cfg if name == primary else _variant_cfg(cfg, name)
        columns = feature_set_columns(cfg, name)
        if name != primary:
            other_input = _model_input_for(z, run_cfg, name)
            select_k(other_input.loc[other_input.index <= window_end], run_cfg)
            log.info("bic_by_k for %s (d=%d) written for information only; K stays %d", name, len(columns), primary_K)
        log.info("running %s (d=%d) into %s", name, len(columns), run_cfg.outputs_tables_dir)
        results[name] = _run_feature_set(z, columns, primary_K, run_cfg)

    refits = [d for d in refit_dates(cfg) if d <= z.index[-1]]
    for name in FEATURE_SETS:
        r = results[name]
        diagnostics.append(
            classifier_diagnostics_row(
                name, r["filtered_labels"], r["smoothed_labels"], refits,
                r["durations"], r["state_counts"], r["chain_table"],
            )
        )
    table = pd.DataFrame(diagnostics, columns=list(CLASSIFIER_DIAGNOSTIC_COLUMNS))
    out = _Path(cfg.outputs_tables_dir) / "classifier_diagnostics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    log.info("classifier_diagnostics.csv written:\n%s", table.to_string(index=False))

    decided = primary_feature_set_decision(table)
    log.info("pre-registered rule selects features.primary = %r (config has %r)", decided, primary)
    if decided != primary:
        log.warning(
            "features.primary in config.toml is %r but the pre-registered rule on the current "
            "diagnostics selects %r; decisions/primary_feature_set.md and config.toml must be "
            "updated together and the outputs regenerated",
            primary, decided,
        )


def _model_input_for(z, cfg: Config, name: str):
    """``model_input`` on a named feature set, without writing over another set's dropped rows."""
    from regime.config import feature_set_columns
    from regime.features import model_input

    return model_input(z, cfg, columns=feature_set_columns(cfg, name))


section_4 = _not_built(4)
section_5 = _not_built(5)
section_6 = _not_built(6)
section_7 = _not_built(7)

SECTIONS: dict[int, Callable[[Config, bool], None]] = {
    1: section_1,
    2: section_2,
    3: section_3,
    4: section_4,
    5: section_5,
    6: section_6,
    7: section_7,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m regime.run")
    parser.add_argument("--section", type=int, choices=sorted(SECTIONS), default=None)
    parser.add_argument("--pull", action="store_true", help="section 1 only: fresh raw pulls")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    sections = [args.section] if args.section is not None else sorted(SECTIONS)
    for n in sections:
        SECTIONS[n](cfg, pull=args.pull if n == 1 else False)


if __name__ == "__main__":
    main()
