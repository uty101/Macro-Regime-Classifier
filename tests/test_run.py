"""Step 1.1: the section registry of python -m regime.run."""

import pytest

from regime.run import SECTIONS


def test_sections_registered_in_order() -> None:
    assert list(SECTIONS) == [1, 2, 3, 4, 5, 6, 7]
    for n, fn in SECTIONS.items():
        assert callable(fn), n


def test_unbuilt_section_raises() -> None:
    from regime.run import _not_built

    # Every section 1 to 7 is built as of section 7, so there is no registered
    # section left to raise. The guard itself is still tested: a section that
    # has not been written raises rather than silently doing nothing.
    from regime.config import load_config

    cfg = load_config()
    with pytest.raises(NotImplementedError, match=r"^section 8 not built$"):
        _not_built(8)(cfg)
