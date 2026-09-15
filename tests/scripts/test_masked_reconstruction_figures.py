"""Figures are drawn from a stored run and from nothing else.

What the drawing is held to: that it reads the run the store wrote, that it draws every figure the
report promises, and that a run stored before the figures were storable is refused with a sentence
rather than a traceback.
"""

from pathlib import Path

import pytest

from emblema.pretraining.adapters.diagnostics.unit_bootstrap import UnitBootstrap
from emblema.pretraining.application.use_cases.assess_reconstruction_run import (
    AssessReconstructionRun,
)
from scripts.masked_reconstruction_assessment import EXAMPLES, RUN, store
from scripts.masked_reconstruction_figures import drawn_from, shape_label
from tests.support.reconstruction_runs import figures, results

assess = AssessReconstructionRun(UnitBootstrap())


@pytest.fixture
def stored(tmp_path: Path) -> Path:
    return store(assess(results()), figures(), tmp_path / "results")


def test_every_figure_the_report_promises_is_drawn_from_the_stored_run(
    stored: Path, tmp_path: Path
) -> None:
    drawn = drawn_from(stored, tmp_path / "figures")

    assert sorted(path.name for path in drawn) == [
        "masked-reconstruction-control-a-diagnostics.png",
        "masked-reconstruction-control-a-loss.png",
        "masked-reconstruction-control-a-windows.png",
    ]
    assert all(path.stat().st_size > 0 for path in drawn)


def test_a_run_that_drew_no_example_still_gets_its_other_figures(
    tmp_path: Path,
) -> None:
    without = store(assess(results()), figures(kinds=()), tmp_path / "results")

    drawn = drawn_from(without, tmp_path / "figures")

    assert [path.name for path in drawn] == [
        "masked-reconstruction-control-a-loss.png",
        "masked-reconstruction-control-a-diagnostics.png",
    ]


def test_a_run_stored_before_the_figures_were_storable_is_refused_in_a_sentence(
    stored: Path, tmp_path: Path
) -> None:
    (stored / EXAMPLES).unlink()

    with pytest.raises(SystemExit, match="running it again"):
        drawn_from(stored, tmp_path / "figures")


def test_a_run_missing_the_baseline_levels_is_refused_the_same_way(
    stored: Path, tmp_path: Path
) -> None:
    lines = (stored / RUN).read_text(encoding="utf-8").splitlines()
    kept = [line for line in lines if not line.startswith(("interpolation_loss", "ridge_loss"))]
    (stored / RUN).write_text("\n".join(kept) + "\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="does not hold what a figure is drawn from"):
        drawn_from(stored, tmp_path / "figures")


def test_the_model_under_a_figure_is_named_by_its_tier_and_its_shape() -> None:
    assert shape_label({"tier": "S", "shape": "192,3,4,768"}) == "tier S, 192 wide, 4 deep"
    assert shape_label({}) == "tier ?,  wide,  deep"
