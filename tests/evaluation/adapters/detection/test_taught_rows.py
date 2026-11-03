import numpy as np
import pytest

from emblema.evaluation.adapters.detection.taught_rows import TaughtRows
from emblema.evaluation.domain.exceptions import InvalidTaughtEventsError
from tests.evaluation.support import labelled, window

FITTING = (window("a", 0, 10.0), window("a", 1, 20.0), window("b", 3, 10.0), window("b", 4, 20.0))


def test_the_taught_windows_are_found_among_the_fitting_rows_by_unit_and_place() -> None:
    taught = TaughtRows.among(
        FITTING,
        (labelled("b", 4, 20.0, 2.0), labelled("a", 1, 20.0, 1.0), labelled("b", 3, 10.0, 2.0)),
    )

    assert taught.positions == (3, 1, 2)
    assert taught.events == (2, 1, 2)
    assert taught.event_count == 2
    assert taught.mask(4).tolist() == [False, True, True, True]


def test_every_event_weighs_one_over_its_windows() -> None:
    taught = TaughtRows(positions=(0, 1, 2), events=(1, 1, 2))

    assert taught.weights().tolist() == [0.5, 0.5, 1.0]
    assert np.allclose(taught.weights().sum(), taught.event_count)


def test_a_taught_window_off_the_fitting_side_or_over_no_event_is_refused() -> None:
    with pytest.raises(InvalidTaughtEventsError, match="not on the fitting side"):
        TaughtRows.among(FITTING, (labelled("c", 9, 10.0, 1.0),))
    with pytest.raises(InvalidTaughtEventsError, match="overlaps an event"):
        TaughtRows.among(FITTING, (labelled("a", 0, 10.0, 0.0),))
    with pytest.raises(InvalidTaughtEventsError, match="past the"):
        TaughtRows(positions=(5,), events=(1,)).mask(4)


def test_rows_without_a_window_named_twice_or_without_an_event_are_refused() -> None:
    with pytest.raises(InvalidTaughtEventsError, match="handed their windows"):
        TaughtRows(positions=(), events=())
    with pytest.raises(InvalidTaughtEventsError, match="named twice"):
        TaughtRows(positions=(1, 1), events=(1, 1))
    with pytest.raises(InvalidTaughtEventsError, match="numbered from one"):
        TaughtRows(positions=(1,), events=(0,))
