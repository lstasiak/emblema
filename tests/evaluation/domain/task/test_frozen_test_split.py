import pytest

from emblema.evaluation.domain.exceptions import InvalidTaskSplitError
from emblema.evaluation.domain.task.frozen_test_split import FrozenTestSplit
from tests.evaluation.support import units


def test_a_frozen_side_says_where_its_units_come_from() -> None:
    side = FrozenTestSplit(units=units("held/1", "held/2"), source="turbofans/test")

    assert side.source == "turbofans/test"
    assert len(side.units) == 2


def test_a_frozen_side_without_units_is_refused() -> None:
    with pytest.raises(InvalidTaskSplitError, match="at least one unit"):
        FrozenTestSplit(units=units(), source="turbofans/test")


@pytest.mark.parametrize("source", ["", "  ", " turbofans/test", "turbofans/test\n"])
def test_a_frozen_side_whose_source_is_blank_or_padded_is_refused(source: str) -> None:
    with pytest.raises(InvalidTaskSplitError, match="source"):
        FrozenTestSplit(units=units("held/1"), source=source)
