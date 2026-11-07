import math

import pytest

from emblema.pretraining.domain.exceptions import InvalidTrainingCorpusError
from emblema.pretraining.domain.training.window_place import WindowPlace


def test_a_window_lies_on_an_axis_over_a_span() -> None:
    place = WindowPlace("ESA-Mission1", 3.0, 4.0)

    assert (place.axis, place.end - place.start) == ("ESA-Mission1", 1.0)


@pytest.mark.parametrize("axis", ["", " a", "a "])
def test_an_axis_without_a_name_is_refused(axis: str) -> None:
    with pytest.raises(InvalidTrainingCorpusError, match="axis"):
        WindowPlace(axis, 0.0, 1.0)


@pytest.mark.parametrize(
    ("start", "end"), [(1.0, 1.0), (2.0, 1.0), (0.0, math.inf), (math.nan, 1.0)]
)
def test_a_span_that_does_not_end_after_it_starts_is_refused(start: float, end: float) -> None:
    with pytest.raises(InvalidTrainingCorpusError, match="end after"):
        WindowPlace("a", start, end)
