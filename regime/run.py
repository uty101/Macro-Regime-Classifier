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


section_2 = _not_built(2)
section_3 = _not_built(3)
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
