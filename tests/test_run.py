"""Step 1.1: the section registry of python -m regime.run. Step 7.3: idempotence and runtime.csv."""

import dataclasses

import pandas as pd
import pytest

from regime.config import load_config
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


def test_run_is_idempotent_for_a_cheap_section(tmp_path) -> None:
    """Section 4 twice into a tmp_path copy of outputs gives identical bytes.

    Section 4 because it is the cheapest section that bootstraps: if a seed
    were not reaching a resampler, this is where it would show. The full
    end-to-end comparison over every section is step 7.3's own run, recorded
    in ``review/section_7.md``; this test is what keeps the property under
    ``pytest``.
    """
    import hashlib

    from regime.run import section_4

    cfg = load_config()
    tables = tmp_path / "tables"
    tables.mkdir()
    variant = dataclasses.replace(cfg, outputs_tables_dir=str(tables))

    section_4(variant)
    first = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(tables.glob("*.csv"))}
    assert "conditional_stats_rules.csv" in first

    section_4(variant)
    second = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(tables.glob("*.csv"))}

    assert first["conditional_stats_rules.csv"] == second["conditional_stats_rules.csv"]
    assert first == second, {k for k in first if first[k] != second.get(k)}


def test_append_runtime_appends_and_never_rewrites(tmp_path) -> None:
    from regime.run import RUNTIME_COLUMNS, append_runtime

    cfg = dataclasses.replace(load_config(), outputs_runtime_log=str(tmp_path / "runtime.csv"))

    append_runtime("20260923T120000Z", 1, 12.5, cfg)
    append_runtime("20260923T120000Z", 2, 3.25, cfg)
    append_runtime("20260923T130000Z", 1, 11.0, cfg)

    written = pd.read_csv(tmp_path / "runtime.csv")
    assert list(written.columns) == list(RUNTIME_COLUMNS) == ["run_started", "section", "seconds"]
    assert len(written) == 3
    assert written["section"].tolist() == [1, 2, 1]
    assert written["seconds"].tolist() == [12.5, 3.25, 11.0]
    # The rows of the first run survive the second run's append.
    assert written["run_started"].nunique() == 2


def test_compare_output_trees_excludes_only_runtime(tmp_path) -> None:
    from regime.run import COMPARISON_COLUMNS, compare_output_trees

    cfg = dataclasses.replace(load_config(), outputs_runtime_log="outputs/tables/runtime.csv")
    run1, run2 = tmp_path / "run1", tmp_path / "run2"
    for root in (run1, run2):
        (root / "tables").mkdir(parents=True)
    (run1 / "tables" / "a.csv").write_text("x,y\n1,2\n")
    (run2 / "tables" / "a.csv").write_text("x,y\n1,2\n")
    (run1 / "tables" / "b.csv").write_text("x\n1\n")
    (run2 / "tables" / "b.csv").write_text("x\n2\n")
    # runtime.csv differs by construction and must not appear in the table.
    (run1 / "tables" / "runtime.csv").write_text("run_started,section,seconds\na,1,1\n")
    (run2 / "tables" / "runtime.csv").write_text("run_started,section,seconds\nb,1,9\n")

    table = compare_output_trees(run1, run2, cfg)

    assert list(table.columns) == list(COMPARISON_COLUMNS)
    assert set(table["path"]) == {"tables/a.csv", "tables/b.csv"}
    by = table.set_index("path")
    assert bool(by.loc["tables/a.csv", "identical"])
    assert not bool(by.loc["tables/b.csv", "identical"])
    assert (by["compared_as"] == "bytes").all()
    assert by.loc["tables/a.csv", "sha_run1"] == by.loc["tables/a.csv", "sha_run2"]


def test_compare_output_trees_reads_pngs_as_pixels(tmp_path) -> None:
    """Two PNGs of the same picture compare identical even if their bytes differ."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from regime.run import compare_output_trees

    cfg = dataclasses.replace(load_config(), outputs_runtime_log="outputs/tables/runtime.csv")
    run1, run2 = tmp_path / "run1", tmp_path / "run2"
    for root, dpi in ((run1, 100), (run2, 100)):
        (root / "charts").mkdir(parents=True)
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.plot([0, 1, 2], [0, 1, 4])
        fig.savefig(root / "charts" / "c.png", dpi=dpi, metadata={"Software": None})
        plt.close(fig)

    table = compare_output_trees(run1, run2, cfg).set_index("path")

    assert table.loc["charts/c.png", "compared_as"] == "pixels"
    assert bool(table.loc["charts/c.png", "identical"])


def test_compare_output_trees_flags_a_file_only_one_run_wrote(tmp_path) -> None:
    from regime.run import compare_output_trees

    cfg = dataclasses.replace(load_config(), outputs_runtime_log="outputs/tables/runtime.csv")
    run1, run2 = tmp_path / "run1", tmp_path / "run2"
    (run1 / "tables").mkdir(parents=True)
    (run2 / "tables").mkdir(parents=True)
    (run1 / "tables" / "only.csv").write_text("x\n1\n")

    table = compare_output_trees(run1, run2, cfg).set_index("path")

    assert table.loc["tables/only.csv", "compared_as"] == "missing"
    assert not bool(table.loc["tables/only.csv", "identical"])
    assert table.loc["tables/only.csv", "sha_run2"] == ""
