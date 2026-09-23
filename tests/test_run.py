"""Step 1.1: the section registry of python -m regime.run."""

import pytest

from regime.run import SECTIONS


def test_sections_registered_in_order() -> None:
    assert list(SECTIONS) == [1, 2, 3, 4, 5, 6, 7]
    for n, fn in SECTIONS.items():
        assert callable(fn), n


def test_unbuilt_section_raises() -> None:
    from regime.config import load_config

    cfg = load_config()
    # Section 6 is built as of this section; 7 is the first that is not.
    with pytest.raises(NotImplementedError, match=r"^section 7 not built$"):
        SECTIONS[7](cfg)
