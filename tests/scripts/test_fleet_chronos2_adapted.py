"""Which fit of an arm answers which cell, and the fits that may not answer them."""

from datetime import date

import pytest

from scripts.fleet_adaptation import FitRecord
from scripts.fleet_chronos2_adapted import assigned
from scripts.fleet_hours import axis_of

AFTER = date(2020, 6, 1)


def fit(name: str, answers: tuple[str, ...], read: tuple[str, ...], arm: str = "a") -> FitRecord:
    return FitRecord(
        name=name,
        arm=arm,
        farm="penmanshiel",
        answers=answers,
        turbines=tuple(f"penmanshiel/{t}" for t in read),
        missions=(),
        share=0.0,
        turbine_series=10,
        mission_series=0,
        until="2020-01-01",
        model="amazon/chronos-2",
        mode="lora",
        rank=8,
        steps=10,
        learning_rate=1e-5,
        schedule="linear",
        context=64,
        batch=8,
        save_every=0,
        seed=1,
    )


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
    with pytest.raises(ValueError, match="starts before"):
        assigned([fit("f0", ("01",), ())], [(5, axis_of("01", date(2019, 6, 1)), 0)])
