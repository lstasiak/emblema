import pytest

from emblema.shared.api.cursor_token import CursorToken
from emblema.shared.kernel.exceptions import InvalidCursorError
from emblema.shared.kernel.paging.cursor import Cursor


def test_a_token_gives_back_the_cursor_it_was_made_of() -> None:
    cursor = Cursor(("boosted_trees", "200", "3"))

    assert CursorToken.decode(CursorToken.encode(cursor)) == cursor


def test_a_token_survives_a_query_string_unchanged() -> None:
    cursor = Cursor(("a/b?c=d&e", "ü"))

    token = CursorToken.encode(cursor)

    assert all(char.isalnum() or char in "-_" for char in token)
    assert CursorToken.decode(token) == cursor


@pytest.mark.parametrize("token", ["", " x", "not base64!", "W10", "e30", "MTIz", "WzFd"])
def test_a_token_this_did_not_issue_is_refused(token: str) -> None:
    with pytest.raises(InvalidCursorError):
        CursorToken.decode(token)
