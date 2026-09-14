"""A run is stored as what it measured and read back into the same assessment, then printed.

No model is trained: the rules read numbers, and numbers are what the tests give them.
"""

from dataclasses import replace
from pathlib import Path

import pytest

from emblema.pretraining.adapters.diagnostics.unit_bootstrap import UnitBootstrap
from emblema.pretraining.application.use_cases.assess_reconstruction_run import (
    AssessReconstructionRun,
)
from emblema.pretraining.domain.assessment.assessment import Assessment
from emblema.pretraining.domain.assessment.results import Results
from emblema.pretraining.domain.mask_kind import MaskKind
from scripts.masked_reconstruction_assessment import (
    INDEX,
    assessment_section,
    find_shorter,
    kinds_table,
    read,
    store,
)
from tests.support.reconstruction_runs import (
    SETTINGS,
    epochs_of,
    half_of,
    learnt_everywhere,
    results,
    tallies,
)

assess = AssessReconstructionRun(UnitBootstrap())


def converged(run: Results) -> Assessment:
    return assess(run, half_of(run))


def test_a_stored_run_reads_back_into_the_same_assessment(tmp_path: Path) -> None:
    run = results(
        unit_tallies=[
            *learnt_everywhere(),
            *tallies(MaskKind.CHANNEL, model=1.3, matched=0.3, apart=True, floor=0.0),
        ]
    )
    found = assess(run)

    directory = store(found, tmp_path)

    assert directory.name == "control-a-20260913-100000"
    assert read(directory) == run
    assert assess(read(directory)).checks == found.checks
    assert sorted(path.name for path in directory.iterdir()) == [
        "checks.csv",
        "epochs.csv",
        "mask_kinds.csv",
        "run.csv",
        "spectrum.csv",
        "tallies.csv",
    ]


def test_a_second_run_of_the_same_second_gets_a_name_of_its_own(tmp_path: Path) -> None:
    found = assess(results())

    first, second = store(found, tmp_path), store(found, tmp_path)

    assert (first.name, second.name) == ("control-a-20260913-100000", "control-a-20260913-100000-2")
    lines = (tmp_path / INDEX).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert lines[0].startswith("run,corpus,code,")
    assert lines[1].split(",")[0] == first.name
    assert lines[2].endswith(",accept")


def test_an_index_of_other_columns_is_not_appended_to(tmp_path: Path) -> None:
    (tmp_path / INDEX).write_text("run,corpus\nold,control-a\n", encoding="utf-8")

    with pytest.raises(ValueError, match="other columns"):
        store(assess(results()), tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == [INDEX]


def test_the_shorter_run_is_the_longest_of_this_configuration_within_half(tmp_path: Path) -> None:
    run = results()
    for epochs, seed in ((2, "1"), (4, "1"), (4, "2")):
        stored = replace(epochs_of(run, epochs), settings={**SETTINGS, "seed": seed})
        store(assess(stored), tmp_path)

    shorter = find_shorter(run, tmp_path)
    of_four = find_shorter(epochs_of(run, 4), tmp_path)

    assert shorter is not None
    assert (shorter.epochs, shorter.settings["seed"]) == (4, "1")
    assert of_four is not None
    assert of_four.epochs == 2
    assert find_shorter(epochs_of(run, 2), tmp_path) is None
    assert find_shorter(run, tmp_path / "nowhere") is None


def test_a_stored_run_that_cannot_be_read_is_an_error_not_a_skip(tmp_path: Path) -> None:
    broken = tmp_path / "control-a-broken"
    broken.mkdir()
    (broken / "run.csv").write_text("key,value\ncorpus,control-a\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not a readable run"):
        find_shorter(results(), tmp_path)


def test_the_tables_lead_with_the_decision_and_the_intervals() -> None:
    found = converged(results())

    section = assessment_section(found)
    kinds = kinds_table(found.summaries)

    assert section.startswith("### Assessment\n\n**accept**")
    assert "- nothing to do" in section
    assert "| token | 2,000 | 20 | 0.0200 | interpolation 0.0400 |" in kinds
    assert "bootstrap over validation units under one draw of the masks" in kinds
