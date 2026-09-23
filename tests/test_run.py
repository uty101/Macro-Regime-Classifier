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


def redirect_outputs(cfg, root):
    """``cfg`` with every path a section writes moved under ``root``.

    The repo's own ``data/processed/``, ``data/interim/``, ``outputs/tables/``,
    ``outputs/charts/`` and ``outputs/regimes/`` are then neither read nor
    written by the run, so a test that builds its own inputs cannot pick up a
    file a previous run of the pipeline happened to leave behind, and cannot
    leave one behind either.

    ``outputs_project1_file`` and ``outputs_external_dir`` are deliberately not
    redirected: the project 1 series are optional (convention 7), nothing
    writes them, and pointing the test at an empty temporary directory would
    force the "absent" branch whatever the repo holds.
    """
    processed, interim = root / "processed", root / "interim"
    tables, charts, regimes = root / "tables", root / "charts", root / "regimes"
    for directory in (processed, interim, tables, charts, regimes):
        directory.mkdir(parents=True, exist_ok=True)
    return dataclasses.replace(
        cfg,
        outputs_interim_dir=str(interim),
        outputs_processed_dir=str(processed),
        outputs_tables_dir=str(tables),
        outputs_charts_dir=str(charts),
        outputs_regimes_dir=str(regimes),
        outputs_asof_panel=str(processed / "asof_panel.parquet"),
        outputs_features_raw=str(processed / "features_raw.parquet"),
        outputs_features_z=str(processed / "features_z.parquet"),
        outputs_dropped_rows=str(processed / "dropped_rows.csv"),
        outputs_primary_k=str(processed / "primary_k.txt"),
        outputs_filtered_probs=str(regimes / "filtered_probs.csv"),
        outputs_smoothed_probs=str(regimes / "smoothed_probs.csv"),
        outputs_gmm_filtered_probs=str(regimes / "gmm_filtered_probs.csv"),
        outputs_regime_labels=str(regimes / "regime_labels.csv"),
        outputs_timing_results=str(tables / "timing_results.csv"),
        outputs_runtime_log=str(tables / "runtime.csv"),
    )


@pytest.fixture(scope="session")
def built_through_section_3(tmp_path_factory):
    """Sections 1, 2 and 3 run into a temporary root, and the ``Config`` that points at it.

    ``data/processed/`` is never committed (convention 14), so on a fresh
    clone every path section 4 reads under it is absent and a test that
    assumes a populated directory fails with ``FileNotFoundError``. The fix is
    to build what is read: section 1 writes the as-of panel from the pinned
    raw pulls that are committed under ``data/raw/`` (``pull=False``, so no
    FRED key and no network), section 2 the features, section 3 the four
    label sources section 4 loads through ``load_label_sources``.

    Session-scoped because sections 1 to 3 take a few minutes together and
    nothing that uses this fixture writes inside the root it returns.
    """
    from regime.run import section_1, section_2, section_3

    cfg = redirect_outputs(load_config(), tmp_path_factory.mktemp("pipeline"))
    section_1(cfg)
    section_2(cfg)
    section_3(cfg)
    return cfg


def test_run_is_idempotent_for_a_cheap_section(tmp_path, built_through_section_3) -> None:
    """Section 4 twice into a tmp_path copy of outputs gives identical bytes.

    Section 4 because it is the cheapest section that bootstraps: if a seed
    were not reaching a resampler, this is where it would show. The full
    end-to-end comparison over every section is step 7.3's own run, recorded
    in ``review/section_7.md``; this test is what keeps the property under
    ``pytest``.

    Its inputs come from ``built_through_section_3``, not from the repo's
    ``data/processed/``, so the test passes on a clean clone.
    """
    import hashlib

    from regime.run import section_4

    tables = tmp_path / "tables"
    tables.mkdir()
    variant = dataclasses.replace(built_through_section_3, outputs_tables_dir=str(tables))

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
