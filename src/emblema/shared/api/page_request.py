"""Which page of a list a request asks for: the parameters every list takes, and their reading."""

from dataclasses import dataclass
from typing import Annotated

from fastapi import Query

from emblema.shared.api.cursor_token import CursorToken
from emblema.shared.kernel.paging.cursor import Cursor

# The two query parameters of every list, declared once so that every list documents them alike.
CursorParameter = Annotated[str | None, Query(description="Where the previous page ended.")]
LimitParameter = Annotated[int | None, Query(ge=1, description="Rows per page.")]


@dataclass(frozen=True, kw_only=True)
class PageRequest:
    """Which page of a list a request asks for.

    A dataclass rather than a model: it is never serialised, and it carries the kernel's cursor.

    Attributes:
        after: Where the previous page ended; ``None`` for the first page.
        limit: How many rows the page holds at most, already bounded by the service.
    """

    after: Cursor | None
    limit: int


class PageRequests:
    """Reads the page a request asks for, by the sizes this process allows.

    The service decides how long a page is when a client does not say, and how long it may be
    when it does; a longer one asked for is cut to the longest, not refused, because the cursor
    of the shorter page still leads on.
    """

    def __init__(self, *, default_size: int, max_size: int) -> None:
        if not 1 <= default_size <= max_size:
            raise ValueError(
                f"a page defaults to between one row and the most, got {default_size} of {max_size}"
            )
        self._default_size = default_size
        self._max_size = max_size

    def read(self, cursor: str | None, limit: int | None) -> PageRequest:
        """The page the parameters ask for.

        Raises:
            InvalidCursorError: If the cursor is not a token this service issued.
        """
        chosen = self._default_size if limit is None else limit
        return PageRequest(
            after=None if cursor is None else CursorToken.decode(cursor),
            limit=min(chosen, self._max_size),
        )
