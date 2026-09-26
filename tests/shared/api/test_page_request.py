import pytest

from emblema.shared.api.cursor_token import CursorToken
from emblema.shared.api.page_request import PageRequest, PageRequests
from emblema.shared.kernel.exceptions import InvalidCursorError
from emblema.shared.kernel.paging.cursor import Cursor

PAGES = PageRequests(default_size=20, max_size=100)


def test_a_page_asked_for_without_parameters_is_the_first_of_the_default_size() -> None:
    assert PAGES.read(None, None) == PageRequest(after=None, limit=20)


def test_a_longer_page_than_the_service_allows_is_cut_to_the_longest() -> None:
    assert PAGES.read(None, 1000).limit == 100


def test_a_cursor_is_read_back_from_its_token() -> None:
    cursor = Cursor(("a", "b"))

    assert PAGES.read(CursorToken.encode(cursor), 5) == PageRequest(after=cursor, limit=5)


def test_a_token_the_service_did_not_issue_is_refused() -> None:
    with pytest.raises(InvalidCursorError):
        PAGES.read("nonsense", None)


@pytest.mark.parametrize(("default_size", "max_size"), [(0, 10), (11, 10)])
def test_a_default_page_lies_between_one_row_and_the_most(default_size: int, max_size: int) -> None:
    with pytest.raises(ValueError, match="defaults to"):
        PageRequests(default_size=default_size, max_size=max_size)
