import pytest

from emblema.shared.kernel.exceptions import InvalidPageError
from emblema.shared.kernel.paging.cursor import Cursor
from emblema.shared.kernel.paging.page import Page


def test_an_empty_page_has_no_next_cursor() -> None:
    with pytest.raises(InvalidPageError):
        Page(items=(), next_cursor=Cursor(("x",)))


def test_a_last_page_holds_its_items_and_no_cursor() -> None:
    page: Page[str] = Page(items=("a", "b"), next_cursor=None)

    assert page.items == ("a", "b")
    assert page.next_cursor is None
