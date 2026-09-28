"""The penalty choices stored and rendered; the fits themselves need the published corpus."""

from itertools import pairwise
from pathlib import Path

import pytest

from scripts.convolution_penalty_report import (
    CHOICES,
    FURTHER,
    IN_FORCE,
    PenaltyChoice,
    main,
    read_choices,
    render,
    write,
)


def choice(chosen: float, *, budget: str = "200", seed: int = 1) -> PenaltyChoice:
    return PenaltyChoice(
        budget=budget,
        seed=seed,
        windows=200,
        positives=28,
        chosen=chosen,
        strongest_in_force=1000.0,
        seconds=12.5,
    )


def test_the_longer_grid_holds_the_one_in_force_and_runs_further_by_the_same_factor() -> None:
    assert FURTHER[: len(IN_FORCE)] == IN_FORCE
    assert list(FURTHER) == sorted(FURTHER)
    ratios = [high / low for low, high in pairwise(FURTHER[len(IN_FORCE) - 1 :])]
    assert all(ratio == pytest.approx(4.64, rel=0.01) for ratio in ratios)


@pytest.mark.parametrize(("chosen", "beyond"), [(215.0, False), (1000.0, True), (4640.0, True)])
def test_a_choice_at_the_strongest_penalty_in_force_or_past_it_is_beyond_the_grid(
    chosen: float, beyond: bool
) -> None:
    assert choice(chosen).beyond_the_grid_in_force is beyond


def test_the_choices_are_written_and_read_back_as_they_were(tmp_path: Path) -> None:
    rows = [choice(215.0, budget="50"), choice(4640.0, budget="all", seed=3)]

    assert write(rows, tmp_path).name == CHOICES
    assert read_choices(tmp_path) == tuple(rows)


def test_the_table_marks_the_choices_past_the_edge() -> None:
    rendered = render([choice(215.0, budget="50"), choice(4640.0)])

    assert "| 50 | 1 | 200 | 28 | 215 | no | 12 |" in rendered
    assert "| 200 | 1 | 200 | 28 | 4640 | yes | 12 |" in rendered
    assert "strongest penalty in force 1000" in rendered


def test_a_run_without_what_it_needs_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="--manifest KEY CHECKSUM are required"):
        main(["--out", str(tmp_path)])
    with pytest.raises(SystemExit, match="nothing to render"):
        main(["--report-only", str(tmp_path)])
