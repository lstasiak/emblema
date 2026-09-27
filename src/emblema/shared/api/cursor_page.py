from collections.abc import Callable
from typing import Self

from pydantic import BaseModel, Field

from emblema.shared.api.cursor_token import CursorToken
from emblema.shared.kernel.paging.page import Page


class CursorPage[T](BaseModel):
    """One page of a list as the API answers it, and the token that opens the next one."""

    items: list[T]
    next_cursor: str | None = Field(
        description="Pass back as `cursor` for the next page; absent on the last one."
    )

    @classmethod
    def of[R](cls, page: Page[R], resource: Callable[[R], T]) -> Self:
        return cls(
            items=[resource(row) for row in page.items],
            next_cursor=None if page.next_cursor is None else CursorToken.encode(page.next_cursor),
        )
