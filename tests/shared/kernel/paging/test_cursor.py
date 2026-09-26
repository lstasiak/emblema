import pytest

from emblema.shared.kernel.exceptions import InvalidCursorError
from emblema.shared.kernel.paging.cursor import Cursor


def test_a_cursor_names_at_least_one_part() -> None:
    with pytest.raises(InvalidCursorError):
        Cursor(())


def test_a_cursor_unpacks_into_as_many_parts_as_its_list_keys_a_row_by() -> None:
    assert Cursor(("boosted_trees", "200", "3")).unpack(3) == ("boosted_trees", "200", "3")


def test_a_cursor_of_another_list_does_not_unpack() -> None:
    with pytest.raises(InvalidCursorError, match="not a cursor of this list"):
        Cursor(("one",)).unpack(2)
