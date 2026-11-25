"""Which fit of an arm answers which cell, and the fits that may not answer them."""

from datetime import date

import pytest

from scripts.fleet_adaptation import FitRecord
from scripts.fleet_chronos2_adapted import assigned, named_as
from scripts.fleet_hours import axis_of
from tests.scripts.test_fleet_adaptation import record

AFTER = date(2020, 6, 1)


def fit(name: str, answers: tuple[str, ...], read: tuple[str, ...], arm: str = "a") -> FitRecord:
    turbines = tuple(f"penmanshiel/{t}" for t in read)
    return record(name=name, answers=answers, turbines=turbines, arm=arm)


def test_each_cell_goes_to_the_one_fit_that_answers_its_turbine() -> None:
    fits = [fit("f0", ("01", "07"), ("02", "04")), fit("f1", ("02", "04"), ("01", "07"))]
    cells = [
        (5, axis_of("01", AFTER), 0),
        (6, axis_of("04", AFTER), 2),
        (7, axis_of("07", AFTER), 1),
    ]

    out = assigned(fits, cells)

    assert out == {"f0": [cells[0], cells[2]], "f1": [cells[1]]}


def test_two_arms_a_turbine_no_fit_answers_or_a_fit_that_read_it_are_refused() -> None:
    cells = [(5, axis_of("01", AFTER), 0)]
    with pytest.raises(ValueError, match="not to one arm"):
        assigned([fit("f0", ("01",), ()), fit("f1", ("02",), (), arm="b")], cells)
    with pytest.raises(ValueError, match="0 fits answer turbine 01"):
        assigned([fit("f0", ("02",), ())], cells)
    with pytest.raises(ValueError, match="2 fits answer turbine 01"):
        assigned([fit("f0", ("01",), ()), fit("f1", ("01",), ())], cells)
    with pytest.raises(ValueError, match="read turbine 01"):
        assigned([fit("f0", ("01",), ("01",))], cells)


def test_adapters_kept_on_the_way_answer_under_a_name_of_their_own_at_a_constant_rate() -> None:
    constant = [record(schedule="constant", arm="s")]

    assert named_as(constant, "checkpoint", None) == "s"
    assert named_as(constant, "checkpoint-8000", "s_8000") == "s_8000"
    with pytest.raises(ValueError, match="constant rate and a name"):
        named_as(constant, "checkpoint-8000", None)
    with pytest.raises(ValueError, match="constant rate and a name"):
        named_as([record(schedule="linear")], "checkpoint-8000", "s_8000")
