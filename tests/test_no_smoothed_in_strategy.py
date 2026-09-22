"""Step 3.8: the timed strategy cannot reach the smoothed labels, enforced on the source text.

The smoothed run is hindsight by construction (step 3.5): one fit on the whole
sample, smoothed over the whole sample. It belongs in the comparison rows of
the timing grid, where it answers "what would perfect regime knowledge have
been worth", and nowhere near the code that decides a weight at t.

Asserting on the text of ``regime/strategy.py`` rather than on its behaviour is
deliberate. A behavioural test would have to guess which call was the leak; the
substring test catches the file name, the ``outputs_smoothed_probs`` config
key, ``run_smoothed_hmm`` and any helper named after it, in one line, and it
fails the moment section 5 reaches for any of them.

The strategy is label-source-agnostic instead: ``run.py`` section 5 loads each
label frame — the smoothed one included, for the hindsight rows — and passes it
in as a plain ``labels`` frame. The strategy never knows which source it holds.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STRATEGY = ROOT / "regime" / "strategy.py"


def test_strategy_never_references_smoothed() -> None:
    source = STRATEGY.read_text(encoding="utf-8")

    assert "smoothed" not in source.lower(), (
        "regime/strategy.py references 'smoothed'. The timed strategy must take "
        "labels as an argument, never load a label source itself — and least of "
        "all the hindsight one. Pass the frame in from run.py section 5."
    )
