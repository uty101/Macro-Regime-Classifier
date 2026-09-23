"""Step 7.4: the README's headline row is the one in timing_results.csv.

The README quotes output rows verbatim. A quoted number that no longer matches
the file it names is a silent lie in the most-read file in the repo, so every
timing row quoted in the prose is parsed back out and compared to the table.
"""

import re
from pathlib import Path

import pandas as pd
import pytest

from regime.config import load_config

README = Path(__file__).resolve().parents[1] / "README.md"
# The decomposition table shares the first six column names, so the header is
# matched on ``p_one_sided`` too — that column exists only in timing_results.csv.
HEADER = "eta  lag  cost_bp       source  sharpe_static  sharpe_timed"
HEADER_ONLY_IN_TIMING_RESULTS = "p_one_sided"

TIMING_COLUMNS = (
    "sharpe_static", "sharpe_timed", "diff", "diff_p05", "diff_p95",
    "p_one_sided", "mean_turnover",
)


def _text() -> str:
    """The README with line endings normalised, so a CRLF checkout parses the same."""
    return README.read_text(encoding="utf-8").replace("\r\n", "\n")


def _headline_blocks() -> list[list[str]]:
    """Every fenced block in the README whose header is a timing_results header."""
    # The optional language tag matters: the README opens with a ```bash block,
    # and a fence pattern that ignores it pairs every later fence off by one.
    blocks = re.findall(r"```[a-z]*\n(.*?)```", _text(), flags=re.S)
    return [
        b.strip("\n").split("\n")
        for b in blocks
        if HEADER in b.split("\n")[0] and HEADER_ONLY_IN_TIMING_RESULTS in b.split("\n")[0]
    ]


def _parse(header: str, row: str) -> dict:
    names, values = header.split(), row.split()
    assert len(names) == len(values), (names, values)
    return dict(zip(names, values))


def test_readme_headline_numbers_match_outputs() -> None:
    cfg = load_config()
    grid = pd.read_csv(cfg.outputs_timing_results)

    blocks = _headline_blocks()
    assert blocks, "the README quotes no timing_results row"

    seen_headline = 0
    for block in blocks:
        header, rows = block[0], block[1:]
        for row in rows:
            quoted = _parse(header, row)
            match = grid.loc[
                (grid["eta"] == float(quoted["eta"]))
                & (grid["lag"] == int(quoted["lag"]))
                & (grid["cost_bp"] == int(quoted["cost_bp"]))
                & (grid["source"] == quoted["source"])
            ]
            assert len(match) == 1, quoted
            match = match.iloc[0]
            for column in TIMING_COLUMNS:
                assert float(quoted[column]) == pytest.approx(float(match[column]), abs=5e-7), (
                    column, quoted, float(match[column])
                )
            assert int(quoted["n_months"]) == int(match["n_months"]), quoted

            seen_headline += (
                float(quoted["eta"]) == cfg.strategy_headline_eta
                and int(quoted["lag"]) == cfg.strategy_headline_lag
                and int(quoted["cost_bp"]) == cfg.strategy_headline_cost_bp
                and quoted["source"] == cfg.strategy_headline_source
            )

    # The headline cell is quoted at least twice: once answering question 3 and
    # once in the "headline row" section PLAN.md step 7.4 requires.
    assert seen_headline >= 2, seen_headline


def test_readme_states_the_null_before_the_questions() -> None:
    """The null is in the first paragraph, ahead of any of the four questions."""
    text = _text()
    first_paragraph = text.split("\n", 1)[1].strip().split("\n\n")[0]

    assert "null" in first_paragraph.lower()
    assert "−0.027" in first_paragraph or "-0.027" in first_paragraph
    assert "refit" in first_paragraph
    assert text.index("The result is a null") < text.index("## The four questions")


def test_readme_never_presents_the_trimmed_value_as_a_result() -> None:
    """The +0.044 trimmed diff is quoted, and named a diagnostic where it is."""
    text = _text()
    assert "0.044294" in text, "the fragility table is quoted"
    assert "fragility diagnostic and is not a result" in text
