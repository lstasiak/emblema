"""What the rule holds out, given units weighed as a published block weighs them."""

from pathlib import Path

import pytest

pytest.importorskip("torch")

from scripts.excursion_report import SIDES, WindowMass, write_rows
from scripts.held_out_units import WeighedUnit, held_out, main, render, sides_of, weighed

pytestmark = pytest.mark.ml


def units(*mean_squares: float) -> list[WeighedUnit]:
    """Units named by their rank, each holding one token of the square it is given."""
    return [
        WeighedUnit(f"u{index:02d}", 1, square)
        for index, square in enumerate(mean_squares, start=1)
    ]


def test_units_are_weighed_by_what_their_windows_hold_heaviest_first() -> None:
    masses = [
        WindowMass("training", "quiet", 0, 2, 2.0, 0, 0.0),
        WindowMass("training", "quiet", 1, 2, 2.0, 0, 0.0),
        WindowMass("validation", "loud", 0, 1, 9.0, 1, 9.0),
    ]

    assert weighed(masses) == [WeighedUnit("loud", 1, 9.0), WeighedUnit("quiet", 4, 1.0)]


def test_the_heavy_units_alternate_and_the_heaviest_stays_on_the_training_side() -> None:
    """A model needs the kind of behaviour it will be asked about, so the heaviest trains."""
    weighed_units = units(50.0, 40.0, 30.0, 20.0, *[0.5] * 16)

    chosen = held_out(weighed_units, fraction=0.2, seed=1)

    assert "u01" not in chosen
    assert {"u02", "u04"} <= set(chosen)
    assert "u03" not in chosen


def test_the_rest_of_the_held_out_side_is_drawn_until_the_share_is_met() -> None:
    weighed_units = units(50.0, 40.0, *[0.5] * 18)

    chosen = held_out(weighed_units, fraction=0.2, seed=1)
    drawn_again = held_out(weighed_units, fraction=0.2, seed=1)
    other_seed = held_out(weighed_units, fraction=0.2, seed=2)

    assert len(chosen) == 4
    assert chosen == drawn_again
    assert chosen != other_seed
    assert list(chosen) == sorted(chosen)


def test_the_units_past_the_line_end_up_on_both_sides() -> None:
    """The point of the rule: no side holds every unit the loss is decided by."""
    weighed_units = units(*[9.0] * 6, *[0.5] * 14)

    sides = sides_of(weighed_units, held_out(weighed_units, fraction=0.2, seed=1))

    for members in sides.values():
        assert any(unit.mean_square > 1.0 for unit in members)


def test_a_line_so_low_that_alternating_fills_the_side_is_refused() -> None:
    weighed_units = units(*[9.0] * 20)

    with pytest.raises(ValueError, match="more than the"):
        held_out(weighed_units, fraction=0.2, seed=1)


def test_the_rule_says_what_it_did_to_each_side(tmp_path: Path) -> None:
    weighed_units = units(50.0, 40.0, *[0.5] * 18)
    chosen = held_out(weighed_units, fraction=0.2, seed=1)

    said = render(weighed_units, chosen, line=1.0)

    assert "| `u01` | 1 | 50.000 | training |" in said
    assert "| `u02` | 1 | 40.000 | held out |" in said
    assert "| training | 16 |" in said
    assert "| validation | 4 |" in said
    assert f"Held out ({len(chosen)} of 20)" in said


def test_the_units_alone_are_printed_for_the_publication_to_take(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_rows(
        tmp_path / SIDES,
        [
            WindowMass("training", f"u{index:02d}", 0, 1, square, 0, 0.0)
            for index, square in enumerate([9.0, 8.0, *[0.5] * 8], start=1)
        ],
    )

    main([str(tmp_path), "--plain"])

    printed = capsys.readouterr().out.split()
    assert len(printed) == 2
    assert "u02" in printed


def test_a_directory_holding_no_weighed_window_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main([str(tmp_path)])
