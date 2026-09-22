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
    from regime.data.fred import FredClient, month_end_market

    log = logging.getLogger("regime")
    if pull:
        client = FredClient(cfg)
        for series_id in cfg.fred_market_series:
            client.pull_market(series_id)
            log.info("pulled market series %s under pull_id %s", series_id, client.pull_id)
        log.info("market pull_id %s (pin as fred.market_pull_id in config.toml)", client.pull_id)

    market_pull_id = cfg.fred_market_pull_id
    if market_pull_id:
        for series_id in cfg.fred_market_series:
            s = month_end_market(series_id, market_pull_id, cfg)
            log.info("%s: %d month-ends, %d NaN", series_id, len(s), int(s.isna().sum()))
    else:
        log.info("fred.market_pull_id not pinned; no month-end series built")


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
