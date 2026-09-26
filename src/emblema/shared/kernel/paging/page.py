from dataclasses import dataclass

from emblema.shared.kernel.exceptions import InvalidPageError
from emblema.shared.kernel.paging.cursor import Cursor


@dataclass(frozen=True)
class Page[T]:
    """One page of a list, and the cursor that opens the next one.

    Invariants: a page with a cursor holds at least one item, because the cursor names the
    last of them.

    Attributes:
        items: The rows of this page, in the list's order.
        next_cursor: Where to continue, or ``None`` where the list ended with this page.
    """

    items: tuple[T, ...]
    next_cursor: Cursor | None

    def __post_init__(self) -> None:
        if self.next_cursor is not None and not self.items:
            raise InvalidPageError("an empty page has no row for a cursor to name")
