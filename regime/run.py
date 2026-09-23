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
import time
from datetime import datetime, timezone
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


def robustness_cfg(cfg: Config, folder: str, **overrides) -> Config:
    """``cfg`` with every output path redirected under ``robustness/<folder>/``, plus ``overrides``.

    Every section 6 variant is driven through here. Nothing in the main
    outputs is touched, and a variant is always a ``dataclasses.replace`` —
    never an edit to ``config.toml`` (section 6's rule, applied early because
    section 3 already runs two feature sets).

    The overrides are applied last, so a variant may replace a redirected path
    as well as a model parameter.
    """
    import dataclasses

    paths = {
        "outputs_tables_dir": f"{cfg.outputs_tables_dir}/robustness/{folder}",
        "outputs_filtered_probs": f"{cfg.outputs_regimes_dir}/robustness/{folder}/filtered_probs.csv",
        "outputs_smoothed_probs": f"{cfg.outputs_regimes_dir}/robustness/{folder}/smoothed_probs.csv",
        "outputs_gmm_filtered_probs": f"{cfg.outputs_regimes_dir}/robustness/{folder}/gmm_filtered_probs.csv",
        "outputs_dropped_rows": f"{cfg.outputs_processed_dir}/dropped_rows_{folder}.csv",
        "outputs_primary_k": f"{cfg.outputs_processed_dir}/primary_k_{folder}.txt",
    }
    paths.update(overrides)
    return dataclasses.replace(cfg, **paths)


def _variant_cfg(cfg: Config, name: str) -> Config:
    """The section 3 feature-set variant: ``robustness_cfg`` under that set's folder."""
    return robustness_cfg(cfg, ROBUSTNESS_DIR_NAME[name])


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


def load_label_sources(cfg: Config) -> dict:
    """The four label frames of ``cfg.strategy_label_sources``, each ``label, assigned`` indexed by date.

    All four come from the primary feature set's main outputs. The three
    probabilistic sources go through ``hard_labels`` (argmax, assigned = max
    probability above ``cfg.hmm_assigned_threshold``); rules labels are always
    assigned (PLAN.md's "Assigned" convention). State numbers already agree
    across the three probabilistic sources through chained anchoring
    (convention 16) and nothing here relabels anything.
    """
    from pathlib import Path as _Path

    import pandas as pd

    from regime.models.hmm import hard_labels

    def _probs(path: str) -> pd.DataFrame:
        frame = pd.read_csv(path, parse_dates=["date"]).set_index("date")
        frame.index = pd.DatetimeIndex(frame.index, name="date")
        return frame

    rules = pd.read_parquet(_Path(cfg.outputs_processed_dir) / "rules_labels.parquet")
    rules_labels_frame = pd.DataFrame(
        {"label": rules["rules_label"].astype("int64"), "assigned": True}, index=rules.index
    )
    rules_labels_frame.index = pd.DatetimeIndex(rules_labels_frame.index, name="date")

    return {
        "hmm_filtered": hard_labels(_probs(cfg.outputs_filtered_probs), cfg),
        "hmm_smoothed": hard_labels(_probs(cfg.outputs_smoothed_probs), cfg),
        "gmm_filtered": hard_labels(_probs(cfg.outputs_gmm_filtered_probs), cfg),
        "rules": rules_labels_frame,
    }


def project1_conditional(cfg: Config, labels: pd.DataFrame):
    """Conditional statistics for the project 1 factors, or ``None`` and a logged skip.

    Convention 7: the project 1 series are optional and nothing may depend on
    them being present. When ``cfg.outputs_project1_file`` is missing this
    logs ``project1: absent, conditional join skipped`` and writes no file.

    When it is present the long (date, factor, ret) frame is pivoted wide,
    ``cfg`` is replaced so ``strategy_factors`` is the project 1 factor tuple,
    and the same ``bootstrap_conditional`` runs against the HMM filtered
    labels on the dates the two have in common.
    """
    import dataclasses
    from pathlib import Path as _Path

    import pandas as pd

    from regime.conditional import add_excess_sharpe, bootstrap_conditional, unconditional_stats
    from regime.data.project1 import load_project1

    log = logging.getLogger("regime")
    p1 = load_project1(cfg.outputs_project1_file)
    if p1.empty:
        log.info("project1: absent, conditional join skipped")
        return None

    wide = p1.pivot(index="date", columns="factor", values="ret").sort_index()
    wide.index = pd.DatetimeIndex(wide.index, name="date")
    wide.columns = [str(c) for c in wide.columns]
    p1_cfg = dataclasses.replace(cfg, strategy_factors=tuple(wide.columns))

    stats, _differences, _nan = bootstrap_conditional(labels, wide, p1_cfg, source="project1")
    stats = add_excess_sharpe(stats, unconditional_stats(labels, wide, p1_cfg))
    tables = _Path(cfg.outputs_tables_dir)
    tables.mkdir(parents=True, exist_ok=True)
    stats.to_csv(tables / "conditional_stats_project1.csv", index=False)
    log.info("conditional_stats_project1.csv written: %d rows over %d factors", len(stats), len(wide.columns))
    return stats


def section_4(cfg: Config, pull: bool = False) -> None:
    """Section 4: conditional factor statistics by regime — steps 4.1 to 4.5.

    The first step of the section that reads a factor return. Every statistic
    is a function of ``join_next_return``'s frame, so the timing convention is
    applied once and cannot be bypassed downstream.
    """
    from pathlib import Path as _Path

    import pandas as pd

    from regime.conditional import (
        add_excess_sharpe,
        bootstrap_conditional,
        bootstrap_refit_split,
        conditional_stats,
        filtered_smoothed_gap,
        join_next_return,
        unassigned_dates,
        unconditional_stats,
    )
    from regime.data.french import load_french
    from regime.models.hmm import refit_dates

    log = logging.getLogger("regime")
    tables = _Path(cfg.outputs_tables_dir)
    tables.mkdir(parents=True, exist_ok=True)

    factors = load_french(cfg.french_pull_id, cfg)
    sources = load_label_sources(cfg)
    log.info("factor file %s: %d months, %s to %s", cfg.french_pull_id, len(factors), factors.index[0].date(), factors.index[-1].date())

    joined = {name: join_next_return(labels, factors, cfg) for name, labels in sources.items()}   # 4.1
    for name in cfg.strategy_label_sources:
        frame = joined[name]
        log.info("%s: %d out-of-sample dates, %s to %s", name, len(frame), frame.index[0].date(), frame.index[-1].date())
    spans = {name: (frame.index[0], frame.index[-1], len(frame)) for name, frame in joined.items()}
    if len(set(spans.values())) != 1:
        log.warning("label sources do not share identical out-of-sample dates: %s", spans)

    for name in cfg.strategy_label_sources:                                                       # 4.2
        stats = conditional_stats(sources[name], factors, cfg)
        stats.to_csv(tables / f"conditional_stats_{name}.csv", index=False)
        missing = unassigned_dates(sources[name], factors, cfg)
        log.info("%s: %d unassigned out-of-sample dates%s", name, len(missing),
                 f" ({', '.join(d.date().isoformat() for d in missing)})" if len(missing) else "")
        log.info("conditional_stats_%s.csv written: %d rows", name, len(stats))

    # The pooled table is written first: every conditional table below carries
    # excess_sharpe beside sharpe, and that needs the unconditional level of
    # each factor over the identical dates (reviewer answer to section 4 Q3,
    # decisions/section_4_review.md).
    pooled = unconditional_stats(sources[cfg.strategy_headline_source], factors, cfg)
    pooled.to_csv(tables / "unconditional_stats.csv", index=False)
    log.info("unconditional_stats.csv written:\n%s", pooled.to_string(index=False))

    refits = refit_dates(cfg)
    split, split_diff = bootstrap_refit_split(sources["hmm_filtered"], factors, refits, cfg)
    split.to_csv(tables / "conditional_stats_refit_split.csv", index=False)
    split_diff.to_csv(tables / "conditional_refit_split_differences.csv", index=False)
    log.info(
        "conditional_stats_refit_split.csv written: %d rows (hmm_filtered only), %d exclude zero; "
        "conditional_refit_split_differences.csv: %d rows, %d exclude zero",
        len(split), int(split["excludes_zero"].sum()),
        len(split_diff), int(split_diff["excludes_zero"].sum()),
    )

    nan_counts = []
    for name in cfg.strategy_label_sources:                                                       # 4.3
        stats, differences, nan_rows = bootstrap_conditional(sources[name], factors, cfg, source=name)
        stats = add_excess_sharpe(stats, pooled)
        stats.to_csv(tables / f"conditional_stats_{name}.csv", index=False)
        differences.to_csv(tables / f"conditional_differences_{name}.csv", index=False)
        nan_counts.append(nan_rows)
        log.info(
            "%s: %d of %d cells exclude zero; %d of %d pairwise differences exclude zero",
            name, int(stats["excludes_zero"].sum()), len(stats),
            int(differences["excludes_zero"].sum()), len(differences),
        )
    nan_table = pd.concat(nan_counts, ignore_index=True)
    nan_table.to_csv(tables / "bootstrap_nan_replications.csv", index=False)
    loud = nan_table.loc[nan_table["share_nan"] > 0.05]
    log.info("bootstrap_nan_replications.csv written: %d rows, max share %.4f, %d above 5%%",
             len(nan_table), float(nan_table["share_nan"].max()), len(loud))
    if len(loud):
        log.warning("cells with over 5%% NaN replications:\n%s", loud.to_string(index=False))

    gap = filtered_smoothed_gap(sources["hmm_filtered"], sources["hmm_smoothed"], factors, cfg)  # 4.4
    gap.to_csv(tables / "filtered_smoothed_gap.csv", index=False)
    log.info(
        "filtered_smoothed_gap.csv written: %d rows, mean absolute gap %.6f",
        len(gap), float(gap["gap"].abs().mean()),
    )

    project1_conditional(cfg, sources["hmm_filtered"])                                            # 4.5


def section_5(cfg: Config, pull: bool = False) -> None:
    """Section 5: the timed strategy — steps 5.1 to 5.5.

    The four label frames are loaded here and passed to ``regime.strategy`` as
    plain ``labels`` arguments. The strategy module never loads a label source
    itself and cannot tell which one it holds (step 3.8), so the hindsight
    labelling can sit in the comparison rows of the grid without ever reaching
    the code that decides a weight at t.

    The headline cell — ``strategy.headline_eta``, ``headline_lag``,
    ``headline_cost_bp``, ``headline_source`` — was fixed in ``config.toml``
    before any of this ran. The other 71 rows are a grid and are logged as one.
    """
    from pathlib import Path as _Path

    from regime.data.french import load_french
    from regime.strategy import (
        fill_timing_gain,
        largest_turnover_months,
        run_timing_grid,
        timing_cells,
        weight_deviation_summary,
    )

    log = logging.getLogger("regime")
    tables = _Path(cfg.outputs_tables_dir)
    tables.mkdir(parents=True, exist_ok=True)

    factors = load_french(cfg.french_pull_id, cfg)
    sources = load_label_sources(cfg)

    deviation = weight_deviation_summary(sources, factors, cfg)                                   # 5.1, 5.2
    deviation.to_csv(tables / "weight_deviation.csv", index=False)
    log.info("weight_deviation.csv written:\n%s", deviation.to_string(index=False))

    grid = run_timing_grid(sources, factors, cfg)                                                 # 5.3, 5.4
    grid = fill_timing_gain(grid, sources, factors, cfg)                                          # 5.5
    grid.to_csv(cfg.outputs_timing_results, index=False)
    log.info("timing_results.csv written: %d rows", len(grid))

    headline = grid.loc[
        (grid["eta"] == cfg.strategy_headline_eta)
        & (grid["lag"] == cfg.strategy_headline_lag)
        & (grid["cost_bp"] == cfg.strategy_headline_cost_bp)
        & (grid["source"] == cfg.strategy_headline_source)
    ]
    log.info("headline cell (fixed in config before the grid was run):\n%s", headline.to_string(index=False))

    decomposition = headline_decomposition(sources, factors, cfg)
    decomposition.to_csv(tables / "timing_headline_decomposition.csv", index=False)
    log.info("timing_headline_decomposition.csv written:\n%s", decomposition.to_string(index=False))

    cells = timing_cells(sources, factors, cfg)
    headline_key = (
        cfg.strategy_headline_eta, cfg.strategy_headline_lag,
        cfg.strategy_headline_cost_bp, cfg.strategy_headline_source,
    )
    top = largest_turnover_months(cells[headline_key]["timed"], 10)
    top.to_csv(tables / "headline_turnover_top10.csv", index=False)
    log.info("headline_turnover_top10.csv written:\n%s", top.to_string(index=False))


DECOMPOSITION_COLUMNS = ("label", "eta", "lag", "cost_bp", "source", "sharpe_static", "sharpe_timed", "diff")


def headline_decomposition(label_frames: dict, factors, cfg: Config):
    """The headline cell beside the same cell at lag 0 and at 0 bp, so the two costs separate.

    The grid has no zero-cost column — ``strategy.cost_bp_grid`` starts at 10 bp
    — so the cost of trading cannot be read off it. These four rows hold the
    source and eta of the headline cell fixed and vary only the lag and the
    cost: lag 1 / 20 bp is the headline, lag 0 / 0 bp is the gross timing
    signal, and the two mixed rows say how much of the gap between them is the
    lag and how much is the trading.
    """
    import pandas as pd

    from regime.strategy import annualised_sharpe, backtest, static_weights, weights

    source, eta = cfg.strategy_headline_source, cfg.strategy_headline_eta
    book = weights(label_frames[source], factors, eta, cfg)
    static_book = static_weights(book.index, cfg)

    variants = (
        ("headline", cfg.strategy_headline_lag, cfg.strategy_headline_cost_bp),
        ("no lag, no cost", 0, 0),
        ("no lag, headline cost", 0, cfg.strategy_headline_cost_bp),
        ("headline lag, no cost", cfg.strategy_headline_lag, 0),
    )
    rows = []
    for label, lag, cost_bp in variants:
        timed = backtest(book, factors, lag, cost_bp, cfg)
        static = backtest(static_book, factors, lag, cost_bp, cfg)
        s_timed = annualised_sharpe(timed["net_ret"].to_numpy(dtype="float64"), cfg.features_ddof)
        s_static = annualised_sharpe(static["net_ret"].to_numpy(dtype="float64"), cfg.features_ddof)
        rows.append((label, eta, lag, cost_bp, source, s_static, s_timed, s_timed - s_static))
    return pd.DataFrame(rows, columns=list(DECOMPOSITION_COLUMNS))


# ---------------------------------------------------------------- section 6

# The two sources every variant that reruns a classifier is judged on. GMM and
# rules rows are not rerun: rules do not depend on K or on the covariance type,
# and the GMM is the comparison model at primary_K only (PLAN.md step 6.1).
ROBUSTNESS_SOURCES = ("hmm_filtered", "hmm_smoothed")

ROBUSTNESS_SUMMARY_COLUMNS = (
    "variant", "sharpe_static", "sharpe_timed", "diff", "diff_p05", "diff_p95",
    "p_one_sided", "mean_turnover", "fallback_share", "n_months",
    "n_pairwise_excl_zero", "n_changes_on_refit_dates",
)
ROBUSTNESS_SUMMARY = "robustness_summary.csv"

K_OUTCOMES = "k_variant_outcomes.csv"
K_OUTCOME_COLUMNS = ("K", "n_free_parameters", "status", "detail")

# The message hmmlearn's ``_utils._validate_covars`` raises for a state
# covariance that is not positive definite, for ``covariance_type="full"`` and
# for ``"tied"``. Matched on the substring because the 'full' form names the
# offending component: "component 4 of 'full' covars must be symmetric,
# positive-definite".
SINGULAR_COVARIANCE_MESSAGE = "positive-definite"


def is_singular_covariance(error: BaseException) -> bool:
    """True for the two ways a rank-deficient state covariance reaches step 6.1.

    Which one is raised is a property of the linear algebra library, not of
    the model. ``forward_filter``'s Cholesky raises
    ``numpy.linalg.LinAlgError``; on another BLAS the eigenvalues hmmlearn
    checks in ``_validate_covars`` come back non-positive first and it raises
    a plain ``ValueError`` before the filter is ever reached. Both mean the
    same fit is rank deficient and both are recorded as
    ``singular_covariance``.

    ``LinAlgError`` is itself a ``ValueError``, so the caller catches
    ``ValueError`` and re-raises anything this rejects: no other exception is
    swallowed.
    """
    import numpy as np

    return isinstance(error, np.linalg.LinAlgError) or (
        isinstance(error, ValueError) and SINGULAR_COVARIANCE_MESSAGE in str(error)
    )


def robustness_dir(cfg: Config):
    """``outputs/tables/robustness/`` — the shared parent of every variant folder."""
    from pathlib import Path as _Path

    path = _Path(cfg.outputs_tables_dir) / "robustness"
    path.mkdir(parents=True, exist_ok=True)
    return path


def variant_classifier(z, columns, K: int, vcfg: Config) -> dict:
    """One variant's expanding and smoothed HMM, written under its own folder.

    The same protocol as the main run's steps 3.4, 3.5 and 3.7: fit at every
    refit date, chain the state numbering (convention 16), filter forward from
    a fresh full-history pass at each t, then one whole-sample smoothed fit
    chained to the last expanding refit.

    Returns ``{"model_input", "params", "labels"}`` where ``labels`` is the two
    ``ROBUSTNESS_SOURCES`` frames.
    """
    from regime.features import model_input
    from regime.models.hmm import hard_labels, run_expanding_hmm, run_smoothed_hmm
    from regime.tables import write_hmm_tables

    x = model_input(z, vcfg, columns=tuple(columns))
    filtered, params = run_expanding_hmm(x, K, vcfg)
    smoothed, _ = run_smoothed_hmm(x, K, vcfg, chain_to=params[-1].means)
    write_hmm_tables(params, list(x.columns), vcfg)
    return {
        "model_input": x,
        "params": params,
        "labels": {
            "hmm_filtered": hard_labels(filtered, vcfg),
            "hmm_smoothed": hard_labels(smoothed, vcfg),
        },
    }


def variant_conditional(labels: dict, factors, vcfg: Config) -> int:
    """Conditional statistics and the block bootstrap for a variant's sources.

    Writes ``unconditional_stats.csv``, ``conditional_stats_<source>.csv`` and
    ``conditional_differences_<source>.csv`` under the variant's tables folder,
    on exactly the dates that variant's own out-of-sample window covers.
    Returns the count of pairwise state differences excluding zero for the
    headline source — the classifier-behaviour number that travels with the
    variant's timing number.
    """
    from pathlib import Path as _Path

    from regime.conditional import add_excess_sharpe, bootstrap_conditional, unconditional_stats

    log = logging.getLogger("regime")
    tables = _Path(vcfg.outputs_tables_dir)
    tables.mkdir(parents=True, exist_ok=True)

    pooled = unconditional_stats(labels[vcfg.strategy_headline_source], factors, vcfg)
    pooled.to_csv(tables / "unconditional_stats.csv", index=False)

    n_pairwise_excl_zero = 0
    for source in vcfg.strategy_label_sources:
        stats, differences, _nan = bootstrap_conditional(labels[source], factors, vcfg, source=source)
        stats = add_excess_sharpe(stats, pooled)
        stats.to_csv(tables / f"conditional_stats_{source}.csv", index=False)
        differences.to_csv(tables / f"conditional_differences_{source}.csv", index=False)
        log.info(
            "%s %s: %d of %d cells and %d of %d pairwise differences exclude zero",
            tables, source, int(stats["excludes_zero"].sum()), len(stats),
            int(differences["excludes_zero"].sum()), len(differences),
        )
        if source == vcfg.strategy_headline_source:
            n_pairwise_excl_zero = int(differences["excludes_zero"].sum())
    return n_pairwise_excl_zero


def headline_row(grid, cfg: Config):
    """The single row of a grid at the headline (eta, lag, cost_bp, source), or a ``KeyError``.

    The headline cell is fixed in ``config.toml`` and is the only cell a
    variant contributes to ``robustness_summary.csv``. A variant that produced
    no such row is a bug, not a missing value, so this raises.
    """
    row = grid.loc[
        (grid["eta"] == cfg.strategy_headline_eta)
        & (grid["lag"] == cfg.strategy_headline_lag)
        & (grid["cost_bp"] == cfg.strategy_headline_cost_bp)
        & (grid["source"] == cfg.strategy_headline_source)
    ]
    if len(row) != 1:
        raise KeyError(f"{len(row)} headline rows in a grid of {len(grid)}; expected exactly 1")
    return row.iloc[0]


def fallback_summary(labels, factors, cfg: Config, eta: float | None = None) -> tuple[int, int, float, float]:
    """``(n_months, n_fallback, fallback_share, mean_abs_deviation)`` for one weight book.

    The denominator is the variant's own out-of-sample decision dates — the
    index ``weights`` builds — not the earning months of a backtest, because a
    fallback is a property of the decision, not of the month it earns.
    ``mean_abs_deviation`` is over every weight of every month, the fallback
    rows included, so it is directly comparable with ``weight_deviation.csv``.
    ``eta`` defaults to ``cfg.strategy_headline_eta``; step 6.4 passes every
    value of the grid.
    """
    import numpy as _np

    from regime.strategy import FALLBACK_NONE, fallback_branches, static_weight, weights

    book = weights(labels, factors, cfg.strategy_headline_eta if eta is None else eta, cfg)
    branches = fallback_branches(book)
    in_fallback = (branches != FALLBACK_NONE).to_numpy(dtype=bool)
    deviation = _np.abs(book.to_numpy(dtype="float64") - static_weight(cfg))
    return int(len(book)), int(in_fallback.sum()), float(in_fallback.mean()), float(deviation.mean())


def variant_summary_row(variant: str, grid, labels: dict, factors, vcfg: Config,
                        n_pairwise_excl_zero: int) -> dict:
    """One row of ``robustness_summary.csv``: a variant's headline-equivalent cell and its classifier.

    The timing number never travels alone. ``fallback_share`` says how much of
    the window the timed book was definitionally the comparator;
    ``n_pairwise_excl_zero`` and ``n_changes_on_refit_dates`` say what the
    classifier underneath it was doing. A variant whose ``diff`` moved because
    its labels stopped moving is not the same finding as one whose ``diff``
    moved because its labels became informative, and those two columns are
    what tells them apart.
    """
    from regime.models.hmm import refit_dates
    from regime.tables import changes_on_refit_dates

    row = headline_row(grid, vcfg)
    n_months, _n_fallback, share, _deviation = fallback_summary(
        labels[vcfg.strategy_headline_source], factors, vcfg
    )
    refits = refit_dates(vcfg)
    _n_changes, n_on_refit, _dates = changes_on_refit_dates(labels["hmm_filtered"], refits)
    return {
        "variant": variant,
        "sharpe_static": float(row["sharpe_static"]),
        "sharpe_timed": float(row["sharpe_timed"]),
        "diff": float(row["diff"]),
        "diff_p05": float(row["diff_p05"]),
        "diff_p95": float(row["diff_p95"]),
        "p_one_sided": float(row["p_one_sided"]),
        "mean_turnover": float(row["mean_turnover"]),
        "fallback_share": share,
        "n_months": int(row["n_months"]),
        "n_pairwise_excl_zero": int(n_pairwise_excl_zero),
        "n_changes_on_refit_dates": int(n_on_refit),
    }


def variant_timing_grid(labels: dict, factors, vcfg: Config, path: str):
    """The variant's timing grid, restricted to ``ROBUSTNESS_SOURCES``, with its bootstrap intervals.

    3 eta x 2 lag x 3 cost x 2 source = 36 rows, written to ``path`` in
    ``outputs/tables/``. The static comparator inside each cell runs on the
    identical earning months by construction: ``timing_cells`` raises if the
    timed and static indices ever differ.
    """
    from regime.strategy import fill_timing_gain, run_timing_grid

    log = logging.getLogger("regime")
    grid = run_timing_grid(labels, factors, vcfg)
    grid = fill_timing_gain(grid, labels, factors, vcfg)
    grid.to_csv(path, index=False)
    log.info("%s written: %d rows", path, len(grid))
    return grid


def append_robustness_summary(rows: list, cfg: Config) -> None:
    """Add rows to ``outputs/tables/robustness/robustness_summary.csv``, replacing any of the same name.

    Written incrementally so each step of section 6 contributes its own
    variants and a rerun of one step does not duplicate them. The file is
    sorted by ``variant`` so it is byte-identical however the steps are
    interleaved.
    """
    import pandas as pd

    log = logging.getLogger("regime")
    path = robustness_dir(cfg) / ROBUSTNESS_SUMMARY
    new = pd.DataFrame(rows, columns=list(ROBUSTNESS_SUMMARY_COLUMNS))
    if path.exists():
        old = pd.read_csv(path)
        old = old.loc[~old["variant"].isin(new["variant"])]
        new = pd.concat([old, new], ignore_index=True)
    new = new.sort_values("variant").reset_index(drop=True)
    new.to_csv(path, index=False)
    log.info("%s written: %d variants\n%s", path, len(new), new.to_string(index=False))


def run_robustness_k(cfg: Config) -> None:
    """Step 6.1 — the whole protocol rerun at every K in ``cfg.hmm_k_grid``.

    For each K: the expanding and smoothed HMM fits (anchoring, restart
    tables, state counts, the chain table) under ``robustness/k<K>/``, the
    conditional statistics and bootstrap for ``hmm_filtered`` and
    ``hmm_smoothed``, and the 36-row timing grid to
    ``outputs/tables/timing_results_k<K>.csv``.

    K = ``primary_K`` is in the grid, so one of these four runs reproduces the
    main run's classifier exactly and is the check that the variant path and
    the main path are the same code.
    """
    from pathlib import Path as _Path

    import pandas as pd

    from regime.config import primary_columns
    from regime.data.french import load_french
    from regime.models.hmm import n_params

    log = logging.getLogger("regime")
    z = pd.read_parquet(cfg.outputs_features_z)
    factors = load_french(cfg.french_pull_id, cfg)
    columns = primary_columns(cfg)
    tables = _Path(cfg.outputs_tables_dir)

    rows, outcomes = [], []
    for K in cfg.hmm_k_grid:
        name = f"k{K}"
        vcfg = robustness_cfg(cfg, name, strategy_label_sources=ROBUSTNESS_SOURCES)
        log.info("step 6.1: K = %d into %s", K, vcfg.outputs_tables_dir)
        try:
            run = variant_classifier(z, columns, K, vcfg)
        except ValueError as error:
            # A rank-deficient state covariance: the filter refuses to evaluate
            # a Gaussian that is not positive definite rather than reaching for
            # a pseudo-inverse. Recorded as an outcome, not swallowed; the
            # decision about what to do with it is decisions/OPEN.md item 1 and
            # is not taken here. Anything else raised by the variant, including
            # any other ValueError, is re-raised untouched.
            if not is_singular_covariance(error):
                raise
            outcomes.append(
                {
                    "K": K,
                    "n_free_parameters": n_params(K, len(columns)),
                    "status": "singular_covariance",
                    "detail": str(error).replace("\n", " "),
                }
            )
            log.error(
                "step 6.1: K = %d could not complete the expanding protocol — %s "
                "(%d free parameters at d = %d). Recorded in k_variant_outcomes.csv "
                "and decisions/OPEN.md; every other K continues.",
                K, error, n_params(K, len(columns)), len(columns),
            )
            continue
        n_excl = variant_conditional(run["labels"], factors, vcfg)
        grid = variant_timing_grid(run["labels"], factors, vcfg, str(tables / f"timing_results_{name}.csv"))
        rows.append(variant_summary_row(name, grid, run["labels"], factors, vcfg, n_excl))
        outcomes.append(
            {"K": K, "n_free_parameters": n_params(K, len(columns)), "status": "completed", "detail": ""}
        )
        log.info("step 6.1 K = %d headline-equivalent row:\n%s", K,
                 pd.DataFrame([rows[-1]]).to_string(index=False))

    table = pd.DataFrame(outcomes, columns=list(K_OUTCOME_COLUMNS))
    table.to_csv(robustness_dir(cfg) / K_OUTCOMES, index=False)
    log.info("%s written:\n%s", K_OUTCOMES, table.to_string(index=False))
    append_robustness_summary(rows, cfg)


def primary_k(cfg: Config) -> int:
    """``primary_K`` as section 3's ``select_k`` wrote it to ``cfg.outputs_primary_k``.

    Read from the file rather than recomputed: K is selected once, on the core
    d = 8 first window, and every section 6 variant that is not itself a sweep
    over K runs at that same K. Recomputing it here would let a variant
    silently change K as well as the thing it is varying.
    """
    from pathlib import Path as _Path

    path = _Path(cfg.outputs_primary_k)
    if not path.exists():
        raise FileNotFoundError(f"{path} is absent; section 3 writes it and must run first")
    return int(path.read_text(encoding="utf-8").strip())


def run_robustness_diag(cfg: Config) -> None:
    """Step 6.2 — the whole protocol at ``primary_K`` with ``covariance_type="diag"``.

    ``dataclasses.replace(cfg, hmm_covariance_type="diag")`` and nothing else.
    ``HMMParams.covars`` stays (K, d, d): hmmlearn's ``covars_`` property
    returns full-shaped matrices for every covariance type, so
    ``forward_filter`` and the anchoring are unchanged and this variant
    differs from the main run only in what the fit was allowed to estimate.
    BIC is not recomputed — K is ``primary_K``, chosen once on the full
    covariance fit, and a diagonal model is not being offered the chance to
    change it.

    Outputs under ``robustness/diag/`` and
    ``outputs/tables/timing_results_diag.csv``.
    """
    from pathlib import Path as _Path

    import pandas as pd

    from regime.config import primary_columns
    from regime.data.french import load_french

    log = logging.getLogger("regime")
    z = pd.read_parquet(cfg.outputs_features_z)
    factors = load_french(cfg.french_pull_id, cfg)
    columns = primary_columns(cfg)
    K = primary_k(cfg)

    vcfg = robustness_cfg(
        cfg, "diag", hmm_covariance_type="diag", strategy_label_sources=ROBUSTNESS_SOURCES
    )
    log.info("step 6.2: covariance_type=diag at K = %d into %s", K, vcfg.outputs_tables_dir)
    run = variant_classifier(z, columns, K, vcfg)
    n_excl = variant_conditional(run["labels"], factors, vcfg)
    grid = variant_timing_grid(
        run["labels"], factors, vcfg, str(_Path(cfg.outputs_tables_dir) / "timing_results_diag.csv")
    )
    row = variant_summary_row("diag", grid, run["labels"], factors, vcfg, n_excl)
    log.info("step 6.2 headline-equivalent row:\n%s", pd.DataFrame([row]).to_string(index=False))
    append_robustness_summary([row], cfg)


def robustness_10feat_cfg(cfg: Config) -> Config:
    """The step 6.3 variant config: 10 features, a 2004-01-31 start, a 2009-12-31 first refit, diagonal.

    Convention 13 fixes all three replacements together, and they belong
    together: a d = 10 model needs a longer first window than 2004 to 2005
    gives, and a full-covariance d = 10 model has 201 parameters at K = 3, so
    the covariance is diagonal. Separating them would produce a variant that
    cannot be fitted, which is exactly what K = 5 does at d = 8
    (``decisions/OPEN.md`` item 1).
    """
    return robustness_cfg(
        cfg,
        "10feat",
        sample_features_from=cfg.sample_robustness_from,
        sample_first_window_end=cfg.hmm_robustness_first_refit,
        hmm_covariance_type=cfg.hmm_robustness_covariance_type,
        strategy_label_sources=ROBUSTNESS_SOURCES,
    )


def robustness_10feat_columns(cfg: Config) -> tuple:
    """``features.core`` + ``features.robustness`` — the ten model-input columns of step 6.3."""
    return tuple(cfg.features_core) + tuple(cfg.features_robustness)


def run_robustness_10feat(cfg: Config) -> None:
    """Step 6.3 — the 10-feature run: `core` + `robustness`, from 2004-01-31, first refit 2009-12-31.

    The z matrix is **not** recomputed. The standardised ``unrate_chg12`` and
    ``breakeven_chg12`` columns are already in ``features_z.parquet`` and their
    standardisation is unchanged — the step 2.3 rule over their own non-NaN
    rows — so this run reads them exactly as the main run reads the core
    eight (convention 3).

    The out-of-sample window for this variant starts at its own first refit,
    2009-12-31, because ``join_next_return`` reads
    ``cfg.sample_first_window_end`` and this variant replaced it. Its
    conditional statistics and its timing grid therefore cover a shorter
    sample than the main run's, and its ``n_months`` in
    ``robustness_summary.csv`` says so.

    Outputs under ``robustness/10feat/`` and
    ``outputs/tables/timing_results_10feat.csv``.
    """
    from pathlib import Path as _Path

    import pandas as pd

    from regime.data.french import load_french

    log = logging.getLogger("regime")
    z = pd.read_parquet(cfg.outputs_features_z)
    factors = load_french(cfg.french_pull_id, cfg)
    K = primary_k(cfg)

    vcfg = robustness_10feat_cfg(cfg)
    columns = robustness_10feat_columns(cfg)
    log.info(
        "step 6.3: d = %d from %s, first refit %s, covariance %s, K = %d into %s",
        len(columns), vcfg.sample_features_from, vcfg.sample_first_window_end,
        vcfg.hmm_covariance_type, K, vcfg.outputs_tables_dir,
    )
    run = variant_classifier(z, columns, K, vcfg)
    log.info(
        "step 6.3 model input: %d rows x %d columns, %s to %s",
        *run["model_input"].shape, run["model_input"].index[0].date(),
        run["model_input"].index[-1].date(),
    )
    n_excl = variant_conditional(run["labels"], factors, vcfg)
    grid = variant_timing_grid(
        run["labels"], factors, vcfg, str(_Path(cfg.outputs_tables_dir) / "timing_results_10feat.csv")
    )
    row = variant_summary_row("10feat", grid, run["labels"], factors, vcfg, n_excl)
    log.info("step 6.3 headline-equivalent row:\n%s", pd.DataFrame([row]).to_string(index=False))
    append_robustness_summary([row], cfg)


MINOBS_FALLBACK_COLUMNS = (
    "min_regime_obs", "source", "eta", "n_months", "n_fallback", "fallback_share", "mean_abs_deviation",
)


def run_robustness_minobs(cfg: Config) -> None:
    """Step 6.4 — the full 72-row grid at every ``min_regime_obs`` in the grid.

    The main run's labels throughout: only ``strategy.min_regime_obs`` moves,
    so every difference between the three files is the effect of that
    threshold and of nothing else. N = 24 is the configured value, so that
    file reproduces ``timing_results.csv`` exactly and is the check that this
    path and section 5's are the same code.

    Beside the ``diff``, and because of the reviewer's answer to section 5 Q2
    (``decisions/section_5_review.md``),
    ``outputs/tables/robustness/minobs_fallback.csv`` carries the number and
    share of out-of-sample months the timed book left at 1/5 and the mean
    absolute deviation of its weights from 1/5, per N, source and eta. A
    larger ``diff`` at N = 12 is read as more trading, not more signal, unless
    the fallback share says otherwise — and these are the columns that say.
    """
    import dataclasses
    from pathlib import Path as _Path

    import pandas as pd

    from regime.data.french import load_french

    log = logging.getLogger("regime")
    factors = load_french(cfg.french_pull_id, cfg)
    sources = load_label_sources(cfg)
    tables = _Path(cfg.outputs_tables_dir)

    fallback_rows = []
    for N in cfg.strategy_min_regime_obs_grid:
        vcfg = dataclasses.replace(cfg, strategy_min_regime_obs=N)
        grid = variant_timing_grid(sources, factors, vcfg, str(tables / f"timing_results_minobs{N}.csv"))
        log.info(
            "step 6.4 min_regime_obs = %d headline-equivalent row:\n%s",
            N, headline_row(grid, vcfg).to_frame().T.to_string(index=False),
        )
        for source in vcfg.strategy_label_sources:
            for eta in vcfg.strategy_eta_grid:
                n_months, n_fallback, share, deviation = fallback_summary(
                    sources[source], factors, vcfg, eta
                )
                fallback_rows.append((N, source, eta, n_months, n_fallback, share, deviation))

    fallback = pd.DataFrame(fallback_rows, columns=list(MINOBS_FALLBACK_COLUMNS))
    fallback.to_csv(robustness_dir(cfg) / "minobs_fallback.csv", index=False)
    log.info(
        "minobs_fallback.csv written: %d rows\n%s",
        len(fallback),
        fallback.loc[
            (fallback["source"] == cfg.strategy_headline_source)
            & (fallback["eta"] == cfg.strategy_headline_eta)
        ].to_string(index=False),
    )


def run_robustness_blocksize(cfg: Config) -> None:
    """Step 6.5 — ``bootstrap.block_size`` over {3, 6, 12}, for the headline cell only.

    Only the resampling changes. The weight book, the static comparator and
    both net-return series are built once, outside the sweep, because none of
    them depends on the block size: what is being varied is how much serial
    dependence the interval is asked to carry, not what it is an interval
    *of*. So every row's ``diff`` is identical by construction and only
    ``diff_p05``, ``diff_p95`` and ``p_one_sided`` move.

    b = 6 is the configured value, so that row reproduces the headline row of
    ``timing_results.csv`` exactly, intervals included.

    Writes ``outputs/tables/timing_results_blocksize.csv`` (``block_size`` +
    the ``timing_results.csv`` columns) and
    ``outputs/tables/conditional_stats_blocksize.csv`` (``block_size`` + the
    nine conditional columns, ``hmm_filtered`` only).
    """
    import dataclasses
    from pathlib import Path as _Path

    import pandas as pd

    from regime.conditional import bootstrap_conditional
    from regime.data.french import load_french
    from regime.strategy import (
        GRID_COLUMNS,
        annualised_sharpe,
        backtest,
        static_weights,
        timing_gain_bootstrap,
        weights,
    )

    log = logging.getLogger("regime")
    factors = load_french(cfg.french_pull_id, cfg)
    sources = load_label_sources(cfg)
    tables = _Path(cfg.outputs_tables_dir)

    source = cfg.strategy_headline_source
    eta, lag, cost_bp = cfg.strategy_headline_eta, cfg.strategy_headline_lag, cfg.strategy_headline_cost_bp
    book = weights(sources[source], factors, eta, cfg)
    timed = backtest(book, factors, lag, cost_bp, cfg)
    static = backtest(static_weights(book.index, cfg), factors, lag, cost_bp, cfg)
    if not timed.index.equals(static.index):
        raise ValueError("the headline cell's timed and static earning months differ")
    sharpe_static = annualised_sharpe(static["net_ret"].to_numpy(dtype="float64"), cfg.features_ddof)
    sharpe_timed = annualised_sharpe(timed["net_ret"].to_numpy(dtype="float64"), cfg.features_ddof)

    timing_rows, stats_frames = [], []
    for block_size in cfg.bootstrap_block_size_grid:
        bcfg = dataclasses.replace(cfg, bootstrap_block_size=block_size)
        gain = timing_gain_bootstrap(static["net_ret"], timed["net_ret"], bcfg)
        timing_rows.append(
            (block_size, eta, lag, cost_bp, source, sharpe_static, sharpe_timed, gain["diff"],
             gain["p05"], gain["p95"], gain["p_one_sided"],
             float(timed["turnover"].mean()), int(len(timed)))
        )
        stats, _differences, _nan = bootstrap_conditional(sources[source], factors, bcfg, source=source)
        stats.insert(0, "block_size", block_size)
        stats_frames.append(stats)

    timing = pd.DataFrame(timing_rows, columns=["block_size"] + list(GRID_COLUMNS))
    timing.to_csv(tables / "timing_results_blocksize.csv", index=False)
    log.info("timing_results_blocksize.csv written:\n%s", timing.to_string(index=False))

    conditional = pd.concat(stats_frames, ignore_index=True)
    conditional.to_csv(tables / "conditional_stats_blocksize.csv", index=False)
    log.info(
        "conditional_stats_blocksize.csv written: %d rows, %s excluding zero per block size",
        len(conditional),
        conditional.groupby("block_size")["excludes_zero"].sum().to_dict(),
    )


def nonprimary_feature_set(cfg: Config) -> str:
    """The feature set ``features.primary`` does not name: ``"core_no_level"`` if it names ``"core"``."""
    from regime.config import FEATURE_SETS

    return next(name for name in FEATURE_SETS if name != cfg.features_primary)


def run_robustness_nonprimary(cfg: Config) -> None:
    """Step 6.6 — the conditional statistics, bootstrap and timing grid of the non-primary feature set.

    The classifier itself is not refitted here. Section 3 runs **both** feature
    sets on every run — that is what keeps
    ``classifier_diagnostics.csv`` from going stale against the
    ``features.primary`` key it justifies — and writes this one's
    ``filtered_probs.csv`` and ``smoothed_probs.csv`` under
    ``robustness/<folder>/`` with the state numbering chained exactly as the
    main run's (convention 16). Step 6.6 reads those and adds the three things
    section 3 does not do, because section 3 never touches a factor return.

    Outputs under ``robustness/nolevel/`` (or ``robustness/level/``, if the
    primary set were ever ``core_no_level``) and
    ``outputs/tables/timing_results_<folder>.csv``.
    """
    from pathlib import Path as _Path

    import pandas as pd

    from regime.config import feature_set_columns
    from regime.data.french import load_french
    from regime.models.hmm import hard_labels

    log = logging.getLogger("regime")
    factors = load_french(cfg.french_pull_id, cfg)
    name = nonprimary_feature_set(cfg)
    folder = ROBUSTNESS_DIR_NAME[name]
    vcfg = robustness_cfg(cfg, folder, strategy_label_sources=ROBUSTNESS_SOURCES)

    def _probs(path: str) -> pd.DataFrame:
        source = _Path(path)
        if not source.exists():
            raise FileNotFoundError(f"{source} is absent; section 3 writes it and must run first")
        frame = pd.read_csv(source, parse_dates=["date"]).set_index("date")
        frame.index = pd.DatetimeIndex(frame.index, name="date")
        return frame

    labels = {
        "hmm_filtered": hard_labels(_probs(vcfg.outputs_filtered_probs), vcfg),
        "hmm_smoothed": hard_labels(_probs(vcfg.outputs_smoothed_probs), vcfg),
    }
    log.info(
        "step 6.6: %s (d = %d) from %s, %d filtered rows",
        name, len(feature_set_columns(cfg, name)), vcfg.outputs_tables_dir,
        len(labels["hmm_filtered"]),
    )

    n_excl = variant_conditional(labels, factors, vcfg)
    grid = variant_timing_grid(
        labels, factors, vcfg, str(_Path(cfg.outputs_tables_dir) / f"timing_results_{folder}.csv")
    )
    row = variant_summary_row(folder, grid, labels, factors, vcfg, n_excl)
    log.info("step 6.6 headline-equivalent row:\n%s", pd.DataFrame([row]).to_string(index=False))
    append_robustness_summary([row], cfg)


def headline_net_returns(cfg: Config, sources: dict, factors):
    """The headline cell's ``(timed, static)`` net-return series, on one index of earning months.

    Exactly the cell ``config.toml`` fixed before section 5 ran — eta
    ``headline_eta``, lag ``headline_lag``, ``headline_cost_bp`` basis points,
    source ``headline_source`` — rebuilt from the same three functions the
    grid used, so step 6.7's statistics sit on the series the grid's ``diff``
    came from and not on a second construction of them.
    """
    from regime.strategy import backtest, static_weights, weights

    source = cfg.strategy_headline_source
    book = weights(sources[source], factors, cfg.strategy_headline_eta, cfg)
    lag, cost_bp = cfg.strategy_headline_lag, cfg.strategy_headline_cost_bp
    timed = backtest(book, factors, lag, cost_bp, cfg)
    static = backtest(static_weights(book.index, cfg), factors, lag, cost_bp, cfg)
    if not timed.index.equals(static.index):
        raise ValueError("the headline cell's timed and static earning months differ")
    return timed["net_ret"], static["net_ret"]


def excluding_zero_pairs(cfg: Config) -> list:
    """The ``(factor, state_a, state_b)`` pairwise differences of the headline source that exclude zero.

    Read from the committed ``conditional_differences_<headline_source>.csv``
    rather than named in code, so step 6.7 cannot go on testing the fragility
    of a difference that section 4 has stopped finding.
    """
    from pathlib import Path as _Path

    import pandas as pd

    path = _Path(cfg.outputs_tables_dir) / f"conditional_differences_{cfg.strategy_headline_source}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} is absent; section 4 writes it and must run first")
    table = pd.read_csv(path)
    hit = table.loc[table["excludes_zero"].astype(bool)]
    return [(row.factor, int(row.state_a), int(row.state_b)) for row in hit.itertuples(index=False)]


def run_robustness_fragility(cfg: Config) -> None:
    """Step 6.7 — how much of each headline statistic one month is carrying.

    Two statistics, two treatments each. The headline timing ``diff`` and the
    one section 4 pairwise state difference whose interval excludes zero are
    each recomputed with every single month dropped in turn, and again with
    the 5% most influential months removed.

    Neither is a result and neither replaces the statistic it is about. What
    they establish is the size of the statistic relative to the influence of
    one month — the question section 5 raised when the headline ``diff``
    changed sign on the removal of 2026-07-31, and section 4 raised when its
    one interval excluding zero turned out to lean on 2009-03.

    Writes ``leave_one_month_out_headline.csv``,
    ``leave_one_month_out_<factor>_<a>_<b>.csv`` and ``fragility_summary.csv``
    under ``outputs/tables/robustness/``.
    """
    import pandas as pd

    from regime.data.french import load_french
    from regime.robustness import (
        FRAGILITY_COLUMNS,
        fragility_row,
        leave_one_month_out,
        leave_one_month_out_pairwise,
        most_influential,
        sharpe_difference,
        trimmed_diff,
        trimmed_pairwise_diff,
    )

    log = logging.getLogger("regime")
    factors = load_french(cfg.french_pull_id, cfg)
    sources = load_label_sources(cfg)
    out = robustness_dir(cfg)

    timed, static = headline_net_returns(cfg, sources, factors)
    full = sharpe_difference(
        timed.to_numpy(dtype="float64"), static.to_numpy(dtype="float64"), cfg
    )
    loo = leave_one_month_out(timed, static, cfg)
    loo.to_csv(out / "leave_one_month_out_headline.csv", index=False)
    trimmed = trimmed_diff(timed, static, cfg=cfg)
    rows = [fragility_row("headline_diff", full, loo, trimmed)]
    log.info(
        "step 6.7 headline diff %.6f over %d months; leaving one out moves it into [%.6f, %.6f], "
        "%d removals flip its sign; trimmed %.6f",
        full, len(loo), float(loo["diff"].min()), float(loo["diff"].max()),
        int(loo["sign_flipped"].sum()), trimmed,
    )
    log.info(
        "the 10 months whose removal moves the headline diff most:\n%s",
        most_influential(loo, full).to_string(index=False),
    )

    pairs = excluding_zero_pairs(cfg)
    log.info("step 6.7: %d pairwise differences of %s exclude zero: %s",
             len(pairs), cfg.strategy_headline_source, pairs)
    labels = sources[cfg.strategy_headline_source]
    for factor, state_a, state_b in pairs:
        pair_loo = leave_one_month_out_pairwise(labels, factors, factor, state_a, state_b, cfg)
        name = f"leave_one_month_out_{factor.lower()}_{state_a}_{state_b}.csv"
        pair_loo.to_csv(out / name, index=False)
        pair_full = _pair_full_value(labels, factors, factor, state_a, state_b, cfg)
        pair_trimmed = trimmed_pairwise_diff(labels, factors, factor, state_a, state_b, cfg)
        rows.append(
            fragility_row(f"{factor.lower()}_{state_a}_{state_b}_sharpe_diff",
                          pair_full, pair_loo, pair_trimmed)
        )
        log.info("%s written: %d months\n%s", name, len(pair_loo),
                 most_influential(pair_loo, pair_full).to_string(index=False))

    summary = pd.DataFrame(rows, columns=list(FRAGILITY_COLUMNS))
    summary.to_csv(out / "fragility_summary.csv", index=False)
    log.info("fragility_summary.csv written:\n%s", summary.to_string(index=False))


def _pair_full_value(labels, factors, factor: str, state_a: int, state_b: int, cfg: Config) -> float:
    """``sharpe(state_b) − sharpe(state_a)`` over every month of the pair, no month dropped."""
    from regime.robustness import _pair_difference, pairwise_months

    months = pairwise_months(labels, factors, factor, state_a, state_b, cfg)
    return _pair_difference(
        months["state"].to_numpy(dtype="int64"), months["ret"].to_numpy(dtype="float64"),
        state_a, state_b, cfg,
    )


def section_6(cfg: Config, pull: bool = False) -> None:
    """Section 6: robustness — steps 6.1 to 6.7.

    Every run here is reported *against* the headline cell of the main run and
    none of them replaces it. The headline cell was fixed in ``config.toml``
    before section 5 computed anything; a variant that happens to look better
    is a variant, and ``robustness_summary.csv`` carries each one's classifier
    behaviour beside its timing number so it can be read as one.
    """
    run_robustness_k(cfg)                                                                  # 6.1
    run_robustness_diag(cfg)                                                               # 6.2
    run_robustness_10feat(cfg)                                                             # 6.3
    run_robustness_minobs(cfg)                                                             # 6.4
    run_robustness_blocksize(cfg)                                                          # 6.5
    run_robustness_nonprimary(cfg)                                                         # 6.6
    run_robustness_fragility(cfg)                                                          # 6.7


# ---------------------------------------------------------------- section 7


def write_charts(cfg: Config) -> dict:
    """Step 7.1: the four PNGs of ``cfg.outputs_charts_dir``, and the legend strings.

    The two heatmaps are written for ``hmm_filtered`` and ``hmm_smoothed``
    because question 4 is the comparison of the two, and a reader who sees the
    filtered heatmap alone has no way to tell how much of what is missing from
    it is the classifier and how much is the factors.
    """
    from pathlib import Path as _Path

    import pandas as pd

    from regime.charts import conditional_sharpe_heatmap, regimes_timeline, timed_vs_static
    from regime.data.french import load_french
    from regime.strategy import backtest, static_weights, weights

    log = logging.getLogger("regime")
    charts = _Path(cfg.outputs_charts_dir)
    charts.mkdir(parents=True, exist_ok=True)
    tables = _Path(cfg.outputs_tables_dir)

    panel_raw = pd.read_parquet(cfg.outputs_features_raw)
    sources = load_label_sources(cfg)
    drift = pd.read_csv(tables / "param_drift.csv")

    regimes_timeline(
        panel_raw, sources["hmm_filtered"], sources["hmm_smoothed"], drift,
        str(charts / "regimes_timeline.png"), cfg,
    )
    legend = _chart_legend(drift, cfg)
    log.info("regimes_timeline.png legend:\n%s", "\n".join(legend[k] for k in sorted(legend)))

    for source, name in (("hmm_filtered", "filtered"), ("hmm_smoothed", "smoothed")):
        stats = pd.read_csv(tables / f"conditional_stats_{source}.csv")
        conditional_sharpe_heatmap(stats, str(charts / f"conditional_sharpe_{name}.png"), cfg)

    factors = load_french(cfg.french_pull_id, cfg)
    lag, cost_bp, eta = cfg.strategy_headline_lag, cfg.strategy_headline_cost_bp, cfg.strategy_headline_eta
    filtered_book = weights(sources["hmm_filtered"], factors, eta, cfg)
    smoothed_book = weights(sources["hmm_smoothed"], factors, eta, cfg)
    backtests = {
        "static": backtest(static_weights(filtered_book.index, cfg), factors, lag, cost_bp, cfg),
        "timed_filtered": backtest(filtered_book, factors, lag, cost_bp, cfg),
        "timed_smoothed": backtest(smoothed_book, factors, lag, cost_bp, cfg),
    }
    timed_vs_static(backtests, str(charts / "timed_vs_static.png"), cfg)
    log.info("charts written to %s", charts)
    return legend


def _chart_legend(drift, cfg: Config) -> dict:
    from regime.charts import state_legend_labels

    return state_legend_labels(drift, cfg)


def write_published_tables(cfg: Config):
    """Step 7.2: ``regime_labels.csv``, then the section 8 completeness check.

    The four label frames come from ``load_label_sources``, the same call
    section 4 and section 5 make, so the published series is the series every
    number in the write-up was computed from and not a second derivation of
    it.
    """
    from pathlib import Path as _Path

    from regime.tables import check_outputs, write_regime_labels

    log = logging.getLogger("regime")
    sources = load_label_sources(cfg)
    labels = write_regime_labels(
        sources["rules"], sources["hmm_filtered"], sources["hmm_smoothed"], sources["gmm_filtered"], cfg
    )
    log.info("regime_labels.csv written: %d rows\n%s", len(labels), labels.head(3).to_string())

    checks = check_outputs(cfg)
    log.info("check_outputs:\n%s", checks.to_string(index=False))
    missing = checks.loc[~checks["present"] | ~checks["columns_ok"]]
    if len(missing):
        log.warning("section 8 tables not in order:\n%s", missing.to_string(index=False))
    checks.to_csv(_Path(cfg.outputs_tables_dir) / "check_outputs.csv", index=False)
    return labels, checks


def section_7(cfg: Config, pull: bool = False) -> None:
    """Section 7: outputs and write-up — steps 7.1 to 7.3.

    Step 7.4 is the README and is not code, so it is not run from here. This
    section regenerates every chart and every table of the kickoff's section 8
    from ``data/processed/``, so a clone that has run sections 1 to 6 can
    rebuild the published outputs without refitting anything.
    """
    write_charts(cfg)                                                                      # 7.1
    write_published_tables(cfg)                                                            # 7.2


RUNTIME_COLUMNS = ("run_started", "section", "seconds")


def append_runtime(run_started: str, section: int, seconds: float, cfg: Config) -> None:
    """One row per section per run, appended to ``cfg.outputs_runtime_log``.

    Appended, never rewritten, for the same reason ``data/raw/manifest.csv``
    is: a run that was slow is evidence, and a file that only ever holds the
    last run cannot show a section getting slower. ``run_started`` is one
    timestamp for the whole run, so the rows of a run group without depending
    on their order.

    This file is the one output excluded from step 7.3's idempotence
    comparison — by construction it differs between two runs, and it is the
    only thing that may.
    """
    import pandas as pd

    from pathlib import Path as _Path

    path = _Path(cfg.outputs_runtime_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([[run_started, section, round(float(seconds), 3)]], columns=list(RUNTIME_COLUMNS))
    row.to_csv(path, mode="a", header=not path.exists(), index=False)


COMPARISON_COLUMNS = ("path", "compared_as", "sha_run1", "sha_run2", "identical")


def compare_output_trees(run1_dir, run2_dir, cfg: Config):
    """Step 7.3's idempotence table: every file under ``outputs/`` from two runs, compared.

    CSV files are compared by the sha256 of their **bytes**; PNG files by
    equality of the decoded pixel array, because a PNG carries a creation
    time in its chunks that no ``metadata`` argument suppresses on every
    matplotlib version, and what has to be reproducible is the picture.

    ``outputs/tables/runtime.csv`` is excluded: it is appended once per
    section per run and differs by construction. Nothing else is excluded.

    Byte equality here is a **same-machine** property. Floating-point output
    moves at the 1e-12 level across Python, BLAS and library versions, so this
    table says two runs of one interpreter on one machine agree; it does not
    say a different machine would produce the same bytes.
    """
    import hashlib
    from pathlib import Path as _Path

    import numpy as np
    import pandas as pd

    run1, run2 = _Path(run1_dir), _Path(run2_dir)
    excluded = _Path(cfg.outputs_runtime_log).name

    def sha(path: _Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    paths = sorted(
        {p.relative_to(run1).as_posix() for p in run1.rglob("*") if p.is_file()}
        | {p.relative_to(run2).as_posix() for p in run2.rglob("*") if p.is_file()}
    )
    rows = []
    for rel in paths:
        if _Path(rel).name == excluded:
            continue
        a, b = run1 / rel, run2 / rel
        if not (a.exists() and b.exists()):
            rows.append({"path": rel, "compared_as": "missing", "sha_run1": sha(a) if a.exists() else "",
                         "sha_run2": sha(b) if b.exists() else "", "identical": False})
            continue
        sha_a, sha_b = sha(a), sha(b)
        if rel.endswith(".png"):
            import matplotlib.image as mpimg

            identical = bool(np.array_equal(mpimg.imread(a), mpimg.imread(b)))
            compared_as = "pixels"
        else:
            identical = sha_a == sha_b
            compared_as = "bytes"
        rows.append({"path": rel, "compared_as": compared_as, "sha_run1": sha_a,
                     "sha_run2": sha_b, "identical": identical})
    return pd.DataFrame(rows, columns=list(COMPARISON_COLUMNS))


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
    log = logging.getLogger("regime")
    cfg = load_config()
    sections = [args.section] if args.section is not None else sorted(SECTIONS)
    run_started = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for n in sections:
        started = time.perf_counter()
        SECTIONS[n](cfg, pull=args.pull if n == 1 else False)
        seconds = time.perf_counter() - started
        append_runtime(run_started, n, seconds, cfg)
        threshold = (
            cfg.run_long_step_timeout_minutes if str(n) in {s.split(".")[0] for s in cfg.run_long_steps}
            else cfg.run_step_timeout_minutes
        )
        log.info("section %d: %.1f s (threshold %d min)", n, seconds, threshold)


if __name__ == "__main__":
    main()
